#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <string>
#include <cfloat>
#include <cmath>
#include <chrono>

using namespace std;

// Help functions for error checking
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

// transformRange_bf16
// CUDA kernel that scales and shifts the input random values from (0,1)
// to the range (a,b) and converts the result to BF16.
// NOTE: this function does NOT perform a boundary check. Please guarantee that exactly numElements threads are launched.
__global__ void transformRange_bf16(const float *in, __nv_bfloat16 *A, float a, float range, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // No boundary check to avoid branch divergence.
    A[i] = __float2bfloat16(in[i] * range + a);
}

// GPU_fill_rand_bf16
// Fills device array A with numElements random BF16 values in the range (a,b).
// The function first generates random float values uniformly on (0,1]
// and then maps them to the desired range while converting to BF16.
void GPU_fill_rand_bf16(__nv_bfloat16 *A, int numElements, unsigned long long seed, float a, float b)
{
    // First, generate random numbers uniformly on (0,1]
    float *temp;
    checkCuda(cudaMallocManaged(&temp, numElements * sizeof(float)));
    
    curandGenerator_t prng;
    checkCurand(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurand(curandSetPseudoRandomGeneratorSeed(prng, seed));
    checkCurand(curandGenerateUniform(prng, temp, numElements));
    checkCurand(curandDestroyGenerator(prng));
    
    // Compute desired range (a,b) and launch a kernel to map from (0,1) -> (a,b) and convert to BF16
    float range = b - a;
    int threadsPerBlock = 256;
    // Ensure that numElements is an exact multiple of threadsPerBlock since the kernel will not check for boundary
    assert(numElements % threadsPerBlock == 0);
    int blocks = numElements / threadsPerBlock;
    transformRange_bf16<<<blocks, threadsPerBlock>>>(temp, A, a, range, numElements);
    checkCuda(cudaDeviceSynchronize());
    
    cudaFree(temp);
}

// runGemmExperiment
// Performs initialization of matrices and runs the GEMM iterations using cuBLAS.
void runGemmExperiment(int dimension, int warmUpPeriod, bool useTensorCores)
{
    int numElements = dimension * dimension;
    // Ensure that numElements is divisible by 256 (required by the kernel launch for GPU_fill_rand_bf16)
    assert(numElements % 256 == 0);

    int m = dimension, n = dimension, k = dimension;
    int lda = m, ldb = k, ldc = m;

    // Create the cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Determine compute type based on the flag.
    cublasComputeType_t computeType = (useTensorCores) ?
                                        CUBLAS_COMPUTE_32F :  // Tensor Core enabled
                                        CUBLAS_COMPUTE_32F_PEDANTIC;  // Tensor Core disabled

    // Create timing events for initialization/allocation
    cudaEvent_t init_start, init_stop;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_stop);

    // Record initialization start timestamp and event
    long long init_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    cudaEventRecord(init_start, 0);

    // Allocate unified memory for all matrices (stored in column-major order)
    __nv_bfloat16 *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, numElements * sizeof(__nv_bfloat16)));
    checkCuda(cudaMallocManaged(&d_B, numElements * sizeof(__nv_bfloat16)));
    checkCuda(cudaMallocManaged(&d_C, numElements * sizeof(__nv_bfloat16)));

    // Initialize Matrices: fill A and B with random values in [0,1] and C with zeros.
    GPU_fill_rand_bf16(d_A, numElements, 0ULL, 0.0f, 1.0f);
    GPU_fill_rand_bf16(d_B, numElements, 0ULL, 0.0f, 1.0f);
    GPU_fill_rand_bf16(d_C, numElements, 0ULL, 0.0f, 0.0f);

    // Record initialization stop timestamp and event
    cudaEventRecord(init_stop, 0);
    cudaEventSynchronize(init_stop);
    long long init_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    float init_elapsed;
    cudaEventElapsedTime(&init_elapsed, init_start, init_stop);

    // Print CSV header with new columns: size,iteration,start_ts,stop_ts,time(ms)
    cout << "size,iteration,start_ts,stop_ts,time(ms)" << endl;
    // Use iteration index = (-warmUpPeriod - 1) for initialization/allocation
    int init_iteration = -warmUpPeriod - 1;
    cout << dimension << "," << init_iteration << "," 
         << init_start_ts << "," << init_stop_ts << "," 
         << init_elapsed << endl;

    // Create timing events for GEMM iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GEMM constants for cuBLAS call
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Timing to track until 4 hours have passed
    const float maxDurationMs = 4.0f * 3600.0f * 1000.0f; // 4 hours in ms
    float cumulativeTime = 0.0f;
    int rep = 0;

    // In this version we always use cuBLAS GEMM.
    // The compute type (and therefore Tensor Core usage) is determined by useTensorCores.
    // Warm-up period (no timing)
    for (int i = 0; i < warmUpPeriod; ++i) {
        // Use cuBLAS GEMM with BF16 operands.
        checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            (const void *)alpha,
            (const void *)d_A, CUDA_R_16BF, lda,
            (const void *)d_B, CUDA_R_16BF, ldb,
            (const void *)beta,
            (void *)d_C, CUDA_R_16BF, ldc,
            computeType, CUBLAS_GEMM_DEFAULT));
    }

    // Timed loop until 4 hours have accumulated
    while (cumulativeTime < maxDurationMs) {
        long long iter_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        cudaEventRecord(start, 0);

        // Use cuBLAS GEMM with BF16 operands.
        checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            (const void *)alpha,
            (const void *)d_A, CUDA_R_16BF, lda,
            (const void *)d_B, CUDA_R_16BF, ldb,
            (const void *)beta,
            (void *)d_C, CUDA_R_16BF, ldc,
            computeType, CUBLAS_GEMM_DEFAULT));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        long long iter_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        cumulativeTime += elapsed;
        cout << dimension << "," << rep << "," 
             << iter_start_ts << "," << iter_stop_ts << "," 
             << elapsed << endl;
        ++rep;
    }

    // Cleanup timing events
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);
}

// main
// Main entry point to the application.
// Usage: <executable> <dimensions> <warm-up period> <device ID> <useTensor (1 for tensor cores, 0 otherwise)>
int main(int argc, char **argv) {
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <dimensions> <repetitions> <warm-up period> <device ID> <useTensor (1/0)>" << endl;
        return -1;
    }

    int dimension = atoi(argv[1]);
    int warmUpPeriod = atoi(argv[2]);
    int deviceID = atoi(argv[3]);
    bool useTensorCores = (atoi(argv[4]) != 0);

    checkCuda(cudaSetDevice(deviceID));

    runGemmExperiment(dimension, warmUpPeriod, useTensorCores);

    return 0;
}
