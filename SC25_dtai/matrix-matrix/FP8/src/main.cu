#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <string>
#include <cfloat>
#include <cmath>

using namespace std;

// Error checking functions
const char *cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

inline cublasStatus_t checkCublas(cublasStatus_t result)
{
    if (result != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS Runtime Error: %s\n", cublasGetErrorString(result));
        assert(result == CUBLAS_STATUS_SUCCESS);
    }
    return result;
}

inline void checkCurand(curandStatus_t status)
{
    if (status != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "CURAND Error: %d\n", status);
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel to convert fp32 values to fp8
__global__ void transformRange_fp8(const float *in, __nv_fp8_e4m3 *A, float a, float range, float scale, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
        // Scale and convert to FP8 E4M3 format
        A[i] = __nv_fp8_e4m3((in[i] * range + a) / scale);
    }
}

// Fill device array with random FP8 values in range [a,b]
void GPU_fill_rand_fp8(__nv_fp8_e4m3 *A, int numElements, unsigned long long seed, float a, float b, float scale)
{
    // Generate random numbers uniformly on (0,1]
    float *temp;
    checkCuda(cudaMallocManaged(&temp, numElements * sizeof(float)));
    
    curandGenerator_t prng;
    checkCurand(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurand(curandSetPseudoRandomGeneratorSeed(prng, seed));
    checkCurand(curandGenerateUniform(prng, temp, numElements));
    checkCurand(curandDestroyGenerator(prng));
    
    // Map to range [a,b] and convert to FP8
    float range = b - a;
    int threadsPerBlock = 256;
    int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    transformRange_fp8<<<blocks, threadsPerBlock>>>(temp, A, a, range, scale, numElements);
    checkCuda(cudaDeviceSynchronize());
    
    cudaFree(temp);
}

// Run FP8 GEMM experiment using cublasLt
void runFP8GemmExperiment(int dimension, int repeats, int warmUpPeriod)
{
    int m = dimension;
    int n = dimension;
    int k = dimension;
    
    // Check for Hopper architecture (compute capability 9.0+)
    cudaDeviceProp prop;
    int device = 0;
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaGetDeviceProperties(&prop, device));
    
    if (prop.major < 9) {
        cerr << "FP8 support requires compute capability 9.0+ (Hopper architecture)" << endl;
        cerr << "Current device: " << prop.name << " with CC " << prop.major << "." << prop.minor << endl;
        return;
    }
    
    // Create cublasLt handle and stream
    cublasLtHandle_t ltHandle;
    checkCublas(cublasLtCreate(&ltHandle));
    
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    
    // Allocate matrices (transposed A for TN format)
    __nv_fp8_e4m3 *d_A, *d_B;
    float *d_C;
    
    // For CUBLAS_OP_T, we need to allocate A with dimensions k x m (transposed)
    checkCuda(cudaMalloc(&d_A, k * m * sizeof(__nv_fp8_e4m3)));
    checkCuda(cudaMalloc(&d_B, k * n * sizeof(__nv_fp8_e4m3)));
    checkCuda(cudaMalloc(&d_C, m * n * sizeof(float)));
    
    // Set scale factors for FP8
    float scaleA = 0.5f;
    float scaleB = 0.5f;
    float *d_scaleA, *d_scaleB;
    
    checkCuda(cudaMalloc(&d_scaleA, sizeof(float)));
    checkCuda(cudaMalloc(&d_scaleB, sizeof(float)));
    
    checkCuda(cudaMemcpyAsync(d_scaleA, &scaleA, sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(d_scaleB, &scaleB, sizeof(float), cudaMemcpyHostToDevice, stream));
    
    // Initialize matrices
    GPU_fill_rand_fp8(d_A, k * m, 0ULL, 0.0f, 1.0f, scaleA);  // Note k * m for transposed A
    GPU_fill_rand_fp8(d_B, k * n, 1ULL, 0.0f, 1.0f, scaleB);
    checkCuda(cudaMemsetAsync(d_C, 0, m * n * sizeof(float), stream));
    checkCuda(cudaStreamSynchronize(stream));
    
    // Create matrix descriptors
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    
    // For transposed A, the logical dimensions are m x k but the physical layout is k x m
    checkCublas(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, k, m, k));
    checkCublas(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, k, n, k));
    checkCublas(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, m));
    
    // Create matmul operation descriptor
    cublasLtMatmulDesc_t matmulDesc;
    checkCublas(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    // Set matrix transform operations: A transposed, B non-transposed (TN format)
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    
    checkCublas(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublas(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    
    // Set scaling factors for FP8 operations
    checkCublas(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA)));
    checkCublas(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB)));
    
    // Create matmul preference with workspace
    cublasLtMatmulPreference_t preference;
    checkCublas(cublasLtMatmulPreferenceCreate(&preference));
    
    // Use 4MB of workspace
    size_t workspaceSize = 4 * 1024 * 1024;
    void* workspace = nullptr;
    checkCuda(cudaMalloc(&workspace, workspaceSize));
    
    checkCublas(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &workspaceSize, sizeof(workspaceSize)));
    
    // Search for algorithms
    cublasLtMatmulHeuristicResult_t heuristicResults[5];
    int returnedAlgoCount = 0;
    
    cublasStatus_t algoStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc,
        preference, 5, heuristicResults, &returnedAlgoCount);
    
    if (algoStatus != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
        cerr << "No suitable FP8 algorithms found. Verify GPU compatibility." << endl;
        return;
    }
    
    // Create timing events
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    
    // GEMM constants
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Run benchmark
    for (int rep = -1 * warmUpPeriod; rep < repeats; rep++)
    {
        if (rep >= 0) {
            checkCuda(cudaEventRecord(start, stream));
        }
        
        // Run FP8 GEMM
        checkCublas(cublasLtMatmul(
            ltHandle, matmulDesc,
            &alpha, d_A, Adesc,
            d_B, Bdesc,
            &beta, d_C, Cdesc,
            d_C, Cdesc,
            &heuristicResults[0].algo,
            workspace, workspaceSize,
            stream));
        
        if (rep >= 0) {
            checkCuda(cudaEventRecord(stop, stream));
            checkCuda(cudaEventSynchronize(stop));
            float elapsed;
            checkCuda(cudaEventElapsedTime(&elapsed, start, stop));
            cout << dimension << "," << rep << "," << elapsed << endl;
        }
    }
    
    // Print final matrix statistics for debugging
    float *h_C = new float[m * n];
    checkCuda(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    delete[] h_C;
    
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtDestroy(ltHandle);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_scaleA);
    cudaFree(d_scaleB);
    cudaFree(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

// Modified main function to only run FP8 code
int main(int argc, char **argv) {
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <dimensions> <repetitions> <warm-up period> <device ID>" << endl;
        return -1;
    }

    int dimension = atoi(argv[1]);
    int repeats = atoi(argv[2]);
    int warmUpPeriod = atoi(argv[3]);
    int deviceID = atoi(argv[4]);

    checkCuda(cudaSetDevice(deviceID));
    
    // Print header for CSV output
    cout << "size,iteration,time(ms)" << endl;
    
    // cout << "Running FP8 E4M3 GEMM experiment..." << endl;
    runFP8GemmExperiment(dimension, repeats, warmUpPeriod);

    return 0;
}