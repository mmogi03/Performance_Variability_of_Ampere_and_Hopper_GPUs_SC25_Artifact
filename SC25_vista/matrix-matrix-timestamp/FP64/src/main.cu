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

// transformRange
// CUDA kernel that scales and shifts the values from (0,1) -> (a,b)
// NOTE: this function does NOT perform a boundary check. Please guarantee that exactly numElements threads are launched.
__global__ void transformRange(double *A, double a, double range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = A[i] * range + a;
}

// GPU_fill_rand_double
// Fills device array A with numElements random double values in the range (a,b).
// Assumes that numElements is an exact multiple of threadsPerBlock.
void GPU_fill_rand_double(double *A, int numElements, unsigned long long seed, double a, double b)
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
    // Ensure that numElements is an exact multiple of threadsPerBlock.
    assert(numElements % threadsPerBlock == 0);
    int blocks = numElements / threadsPerBlock;
    transformRange<<<blocks, threadsPerBlock>>>(A, a, range);
    checkCuda(cudaDeviceSynchronize());
}

// runGemmExperiment
// Performs initialization of matrices and runs the GEMM iterations using cuBLAS.
void runGemmExperiment(int dimension, int repeats, int warmUpPeriod, bool useTensorCores)
{
    int numElements = dimension * dimension;
    // Ensure that numElements is divisible by 256 (required by GPU_fill_rand_double).
    assert(numElements % 256 == 0);

    int m = dimension, n = dimension, k = dimension;
    int lda = m, ldb = k, ldc = m;

    // Create the cuBLAS handle.
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Determine compute type based on the flag.
    cublasComputeType_t computeType;
    if (useTensorCores)
    {
        computeType = CUBLAS_COMPUTE_64F; // Tensor Core enabled for FP64 (if available)
    }
    else
    {
        computeType = CUBLAS_COMPUTE_64F_PEDANTIC; // Tensor Core disabled for FP64
    }

    // Create timing events for initialization/allocation.
    cudaEvent_t init_start, init_stop;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_stop);

    // Record initialization start timestamp and event.
    long long init_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    cudaEventRecord(init_start, 0);

    // Allocate unified memory for all matrices (stored in column-major order).
    double *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, numElements * sizeof(double)));
    checkCuda(cudaMallocManaged(&d_B, numElements * sizeof(double)));
    checkCuda(cudaMallocManaged(&d_C, numElements * sizeof(double)));

    // Initialize Matrices: fill A and B with random values in [0,1] and C with zeros.
    GPU_fill_rand_double(d_A, numElements, 0ULL, 0.0, 1.0);
    GPU_fill_rand_double(d_B, numElements, 0ULL, 0.0, 1.0);
    GPU_fill_rand_double(d_C, numElements, 0ULL, 0.0, 0.0);

    // Record initialization stop timestamp and event.
    cudaEventRecord(init_stop, 0);
    cudaEventSynchronize(init_stop);
    long long init_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    float init_elapsed;
    cudaEventElapsedTime(&init_elapsed, init_start, init_stop);

    // Print CSV header with columns: size,iteration,start_ts,stop_ts,time(ms)
    cout << "size,iteration,start_ts,stop_ts,time(ms)" << endl;
    // Use iteration index = (-warmUpPeriod - 1) for initialization/allocation.
    int init_iteration = -warmUpPeriod - 1;
    cout << dimension << "," << init_iteration << ","
         << init_start_ts << "," << init_stop_ts << ","
         << init_elapsed << endl;

    // Create timing events for GEMM iterations.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GEMM constants for cuBLAS call.
    const double alf = 1.0;
    const double bet = 0.0;
    const double *alpha = &alf;
    const double *beta = &bet;

    // Run GEMM iterations, including warm-up iterations.
    for (int rep = -warmUpPeriod; rep < repeats; rep++)
    {
        // Record iteration start timestamp and event.
        long long iter_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        cudaEventRecord(start, 0);

        // Use cuBLAS GEMM with FP64 operands.
        checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  (const void *)alpha,
                                  (const void *)d_A, CUDA_R_64F, lda,
                                  (const void *)d_B, CUDA_R_64F, ldb,
                                  (const void *)beta,
                                  (void *)d_C, CUDA_R_64F, ldc,
                                  computeType, CUBLAS_GEMM_DEFAULT));

        // Record iteration stop timestamp and event.
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        long long iter_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        cout << dimension << "," << rep << ","
             << iter_start_ts << "," << iter_stop_ts << ","
             << elapsed << endl;
    }

    // Cleanup timing events.
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory.
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

    runGemmExperiment(dimension, repeats, warmUpPeriod, useTensorCores);

    return 0;
}
