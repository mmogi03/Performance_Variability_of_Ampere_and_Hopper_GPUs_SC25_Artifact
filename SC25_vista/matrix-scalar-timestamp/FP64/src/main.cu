#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
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

// transformRange_fp64
// CUDA kernel that scales and shifts the values from (0,1) -> (a,b).
// NOTE: this function does NOT perform a boundary check.
__global__ void transformRange_fp64(double *A, double a, double range, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = A[i] * range + a;
}

// GPU_fill_rand_fp64
// Fills device array A with numElements random FP64 values in the range (a,b).
void GPU_fill_rand_fp64(double *A, int numElements, unsigned long long seed, double a, double b)
{
    curandGenerator_t prng;
    checkCurand(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurand(curandSetPseudoRandomGeneratorSeed(prng, seed));
    checkCurand(curandGenerateUniformDouble(prng, A, numElements));
    checkCurand(curandDestroyGenerator(prng));

    double range = b - a;
    int threadsPerBlock = 256;
    assert(numElements % threadsPerBlock == 0);
    int blocks = numElements / threadsPerBlock;
    transformRange_fp64<<<blocks, threadsPerBlock>>>(A, a, range, numElements);
    checkCuda(cudaDeviceSynchronize());
}

// runScalarExperiment
// Performs initialization of the vector and runs the scalar scaling operations.
void runScalarExperiment(int size, int repeats, int warmUpPeriod)
{
    // Timing for initialization/allocation.
    cudaEvent_t init_start, init_stop;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_stop);
    long long init_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    cudaEventRecord(init_start, 0);

    double *d_A;
    checkCuda(cudaMallocManaged(&d_A, size * sizeof(double)));

    // Initialize vector: values in [0.0, 1.0].
    GPU_fill_rand_fp64(d_A, size, 0ULL, 0.0, 1.0);

    cudaEventRecord(init_stop, 0);
    cudaEventSynchronize(init_stop);
    long long init_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    float init_elapsed;
    cudaEventElapsedTime(&init_elapsed, init_start, init_stop);

    cout << "size,iteration,start_ts,stop_ts,time(ms)" << endl;
    int init_iteration = -warmUpPeriod - 1;
    cout << size << "," << init_iteration << ","
         << init_start_ts << "," << init_stop_ts << ","
         << init_elapsed << endl;

    // Timing for scalar iterations.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    double alpha = 2.0, beta = 0.5;
    int incx = 1;

    for (int rep = -warmUpPeriod; rep < repeats; rep++)
    {
        long long iter_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        cudaEventRecord(start, 0);

        checkCublas(cublasDscal(handle, size, &alpha, d_A, incx));
        checkCublas(cublasDscal(handle, size, &beta, d_A, incx));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        long long iter_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);

        // Log every iteration (including warm-up iterations)
        cout << size << "," << rep << ","
             << iter_start_ts << "," << iter_stop_ts << ","
             << elapsed << endl;
    }

    cudaEventDestroy(init_start);
    cudaEventDestroy(init_stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cublasDestroy(handle);
}

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
    runScalarExperiment(size, repeats, warmUpPeriod);

    return 0;
}
