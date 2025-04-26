#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <string>
#include <cfloat>
#include <cmath>

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

// transformRange_fp16
// CUDA kernel that scales and shifts the values from (0,1) -> (a,b)
// and converts the result to FP16.
// NOTE: this function does NOT perform a boundary check. Please guarantee that exactly numElements threads are launched.
__global__ void transformRange_fp16(const float *in, __half *A, float a, float range, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = __float2half(in[i] * range + a);
}

// GPU_fill_rand_fp16
// Fills device array A with numElements random FP16 values in the range (a,b).
// The function first generates random float values uniformly on (0,1]
// and then maps them to the desired range while converting to FP16.
void GPU_fill_rand_fp16(__half *A, int numElements, unsigned long long seed, float a, float b)
{
    // First, generate random numbers uniformly on (0,1]
    float *temp;
    checkCuda(cudaMallocManaged(&temp, numElements * sizeof(float)));
    
    curandGenerator_t prng;
    checkCurand(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurand(curandSetPseudoRandomGeneratorSeed(prng, seed));
    checkCurand(curandGenerateUniform(prng, temp, numElements));
    checkCurand(curandDestroyGenerator(prng));
    
    // Compute desired range (a,b) and launch a kernel to map from (0,1) -> (a,b) and convert to FP16
    float range = b - a;
    int threadsPerBlock = 256;
    // Ensure that numElements is an exact multiple of threadsPerBlock since the kernel will not check for boundary
    assert(numElements % threadsPerBlock == 0);
    int blocks = numElements / threadsPerBlock;
    transformRange_fp16<<<blocks, threadsPerBlock>>>(temp, A, a, range, numElements);
    checkCuda(cudaDeviceSynchronize());
    
    cudaFree(temp);
}

// runGemmExperiment
// Performs initialization of matrices and runs the GEMM iterations using cuBLAS.
// The compute type is selected based on the user parameter useTensorCores.
// If useTensorCores is true, we use CUBLAS_COMPUTE_16F (which enables Tensor Core optimizations).
// Otherwise, we use CUBLAS_COMPUTE_16F_PEDANTIC (which disables them).
void runGemmExperiment(int dimension, int repeats, int warmUpPeriod, bool useTensorCores)
{
    int numElements = dimension * dimension;
    // Ensure that numElements is divisible by 256 (required by the kernel launch for GPU_fill_rand_fp16)
    assert(numElements % 256 == 0);

    int m = dimension, n = dimension, k = dimension;
    int lda = m, ldb = k, ldc = m;

    // Create the cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Determine compute type based on the flag.
    cublasComputeType_t computeType;
    if (useTensorCores)
    {
        computeType = CUBLAS_COMPUTE_16F;           // Tensor Core enabled for FP16
    }
    else
    {
        computeType = CUBLAS_COMPUTE_16F_PEDANTIC;    // Tensor Core disabled for FP16
    }

    // Allocate unified memory for all matrices (stored in column-major order)
    __half *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, numElements * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_B, numElements * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_C, numElements * sizeof(__half)));

    // Initialize Matrices: fill A and B with random values in [0,1] and C with zeros.
    GPU_fill_rand_fp16(d_A, numElements, 0ULL, 0.0f, 1.0f);
    GPU_fill_rand_fp16(d_B, numElements, 0ULL, 0.0f, 1.0f);
    GPU_fill_rand_fp16(d_C, numElements, 0ULL, 0.0f, 0.0f);

    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GEMM constants for cuBLAS call
    const __half alf = __float2half(1.0f);
    const __half bet = __float2half(0.0f);
    const __half *alpha = &alf;
    const __half *beta = &bet;

    // In this version we always use cuBLAS GEMM.
    // The compute type (and therefore Tensor Core usage) is determined by useTensorCores.
    for (int rep = -1 * warmUpPeriod; rep < repeats; rep++)
    {
        if (rep >= 0)
        {
            cudaEventRecord(start, 0);
        }

        // Use cuBLAS GEMM with FP16 operands.
        checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 (const void *)alpha,
                                 (const void *)d_A, CUDA_R_16F, lda,
                                 (const void *)d_B, CUDA_R_16F, ldb,
                                 (const void *)beta,
                                 (void *)d_C, CUDA_R_16F, ldc,
                                 computeType, CUBLAS_GEMM_DEFAULT));

        if (rep >= 0)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            cout << dimension << "," << rep << "," << elapsed << endl;
        }
    }
    
    // Destroy timing events
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
// Usage: <executable> <dimensions> <repetitions> <warm-up period> <device ID> <useTensor (1 for tensor cores, 0 otherwise)>
int main(int argc, char **argv) {
    if (argc < 6)
    {
        cerr << "Usage: " << argv[0] << " <dimensions> <repetitions> <warm-up period> <device ID> <useTensor (1/0)>" << endl;
        return -1;
    }

    int dimension = atoi(argv[1]);
    int repeats = atoi(argv[2]);
    int warmUpPeriod = atoi(argv[3]);
    int deviceID = atoi(argv[4]);
    bool useTensorCores = (atoi(argv[5]) != 0);

    checkCuda(cudaSetDevice(deviceID));

    cout << "size,iteration,time(ms)" << endl;
    runGemmExperiment(dimension, repeats, warmUpPeriod, useTensorCores);

    return 0;
}
