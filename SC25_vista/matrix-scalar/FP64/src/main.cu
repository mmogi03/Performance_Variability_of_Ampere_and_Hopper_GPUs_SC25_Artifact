#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
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

// transformRange_fp64
// CUDA kernel that scales and shifts the values from (0,1) -> (a,b)
// NOTE: this function does NOT perform a boundary check. Please guarantee that exactly numElements threads are launched.
__global__ void transformRange_fp64(double *A, double a, double range, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = A[i] * range + a;
}

// GPU_fill_rand_fp64
// Fills device array A with numElements random FP64 values in the range (a,b)
// Assumes that numElements is an exact multiple of threadsPerBlock.
void GPU_fill_rand_fp64(double *A, int numElements, unsigned long long seed, double a, double b)
{
    // First, generate random numbers uniformly on (0,1]
    curandGenerator_t prng;
    checkCurand(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurand(curandSetPseudoRandomGeneratorSeed(prng, seed));
    checkCurand(curandGenerateUniformDouble(prng, A, numElements));
    checkCurand(curandDestroyGenerator(prng));

    // Compute desired range (a,b) and launch a kernel to map from (0,1) -> (a,b)
    double range = b - a;
    int threadsPerBlock = 256;
    // Ensure that numElements is an exact multiple of threadsPerBlock
    assert(numElements % threadsPerBlock == 0);
    int blocks = numElements / threadsPerBlock;
    transformRange_fp64<<<blocks, threadsPerBlock>>>(A, a, range, numElements);
    checkCuda(cudaDeviceSynchronize());
}

// runScalarExperiment
// Performs initialization of the vector and runs the scalar scaling operations.
void runScalarExperiment(int size, int repeats, int warmUpPeriod)
{
    // Allocate unified memory for the vector
    double *d_A;
    checkCuda(cudaMallocManaged(&d_A, size * sizeof(double)));

    // Initialize vector
    // Values in [0.0, 1.0]
    GPU_fill_rand_fp64(d_A, size, 0ULL, 0.0, 1.0);

    // Create the cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Create timing events if times are requested upon launch of application
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Scalar constants
    // alpha = 2.0 (for the first scaling), beta = 1.0 (for the second scaling)
    double alpha = 2.0;
    double beta  = 0.5;
    int incx = 1;

    // Iteration loop: each iteration performs two scaling operations (vector-scalar multiplications)
    // NOTE: the warm-up period is performed before logging timings.
    for (int rep = -1 * warmUpPeriod; rep < repeats; rep++)
    {
        // Begin recording iteration times after warm-up period
        if (rep >= 0)
        {
            cudaEventRecord(start, 0);
        }

        // Scalar operation: x = alpha * x
        checkCublas(cublasDscal(handle, size, &alpha, d_A, incx));

        // Scalar operation: x = beta * x
        checkCublas(cublasDscal(handle, size, &beta, d_A, incx));

        if (rep >= 0)
        {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            cout << size << "," << rep << "," << elapsed << endl;
        }
    }

    // Destroy timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory
    cudaFree(d_A);

    cublasDestroy(handle);
}

// main
// Main entry point to the application
// Sets the device and then runs the measured experiment.
int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <size> <repetitions> <warm-up period> <device ID>" << endl;
        return -1;
    }

    int size = atoi(argv[1]);
    int repeats = atoi(argv[2]);
    int warmUpPeriod = atoi(argv[3]);
    int deviceID = atoi(argv[4]);

    checkCuda(cudaSetDevice(deviceID));

    cout << "size,iteration,time(ms)" << endl;
    runScalarExperiment(size, repeats, warmUpPeriod);

    return 0;
}
