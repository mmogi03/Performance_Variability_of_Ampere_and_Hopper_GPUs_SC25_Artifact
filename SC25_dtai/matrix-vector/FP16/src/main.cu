#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdlib>
#include <string>
#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>

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
// and converts the result to __half.
// NOTE: this function does NOT perform a boundary check. Please guarantee that exactly numElements threads are launched.
__global__ void transformRange_fp16(const float *in, __half *A, float a, float range, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = __float2half(in[i] * range + a);
}

// GPU_fill_rand_fp16
// Fills device array A with numElements random FP16 values in the range (a,b)
// Assumes that numElements is an exact multiple of threadsPerBlock.
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

    // Compute desired range (a,b) and launch a kernel to map from (0,1) -> (a,b)
    float range = b - a;
    int threadsPerBlock = 256;
    assert(numElements % threadsPerBlock == 0);
    int blocks = numElements / threadsPerBlock;
    transformRange_fp16<<<blocks, threadsPerBlock>>>(temp, A, a, range, numElements);
    checkCuda(cudaDeviceSynchronize());
    cudaFree(temp);
}

// runGemvExperiment
// Performs initialization of the matrix and vector and runs the batched matrix-vector operations.
// The elapsed time for these operations is recorded.
void runGemvExperiment(int dimension, int repeats, int warmUpPeriod)
{
    int m = dimension, n = dimension;
    int numElementsMatrix = m * n;
    int numElementsVector = m;  // assuming square matrix and vector length m

    // Create the cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Allocate unified memory for matrix A and vectors x and y (FP16)
    __half *d_A, *d_x, *d_y;
    checkCuda(cudaMallocManaged(&d_A, numElementsMatrix * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_x, numElementsVector * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_y, numElementsVector * sizeof(__half)));

    // Allocate unified memory for arrays of pointers (for batched operation, batchCount = 1)
    __half **d_Aarray, **d_xarray, **d_yarray;
    checkCuda(cudaMallocManaged(&d_Aarray, sizeof(__half*)));
    checkCuda(cudaMallocManaged(&d_xarray, sizeof(__half*)));
    checkCuda(cudaMallocManaged(&d_yarray, sizeof(__half*)));

    *d_Aarray = d_A;
    *d_xarray = d_x;
    *d_yarray = d_y;

    // Initialize data arrays with random values in [0.0, 1.0]
    GPU_fill_rand_fp16(d_A, numElementsMatrix, 0ULL, 0.0f, 1.0f);
    GPU_fill_rand_fp16(d_x, numElementsVector, 0ULL, 0.0f, 1.0f);
    // Initialize y to zeros
    for (int i = 0; i < numElementsVector; i++)
    {
        d_y[i] = __float2half(0.0f);
    }

    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GEMV constants: alpha = 1.0, beta = 0
    float alf = 1.0f;
    float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;
    int lda = m;
    int incx = 1, incy = 1;
    int batchCount = 1;

    // Iteration loop: each iteration performs GEMV operation
    // NOTE: the warm-up period is performed before logging timings.
    for (int rep = -1 * warmUpPeriod; rep < repeats; rep++)
    {
        if (rep >= 0)
        {
            cudaEventRecord(start, 0);
        }

        // batched GEMV: y = alpha * A * x
        checkCublas(cublasHSHgemvBatched(handle,
            CUBLAS_OP_N, m, n,
            alpha,
            (const __half *const *)d_Aarray, lda,
            (const __half *const *)d_xarray, incx,
            beta,
            (__half *const *)d_yarray, incy,
            batchCount));

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

    // Free device memory
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_x));
    checkCuda(cudaFree(d_y));
    checkCuda(cudaFree(d_Aarray));
    checkCuda(cudaFree(d_xarray));
    checkCuda(cudaFree(d_yarray));

    cublasDestroy(handle);
}

// main
// Main entry point to the application
// Sets the device and then runs the measured experiment.
int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <dimension> <repetitions> <warm-up period> <device ID>" << endl;
        return -1;
    }

    int dimension = atoi(argv[1]);
    int repeats = atoi(argv[2]);
    int warmUpPeriod = atoi(argv[3]);
    int deviceID = atoi(argv[4]);

    checkCuda(cudaSetDevice(deviceID));

    cout << "dimension,iteration,time(ms)" << endl;
    runGemvExperiment(dimension, repeats, warmUpPeriod);

    return 0;
}