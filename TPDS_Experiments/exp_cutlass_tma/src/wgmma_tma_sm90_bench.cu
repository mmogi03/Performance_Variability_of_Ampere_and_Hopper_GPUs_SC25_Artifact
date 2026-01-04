/***************************************************************************************************
 * SM90 WGMMA GEMM micro-benchmark (TMA path)
 * - Based closely on: examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
 * - Adds CSV logging with stage timestamps + CUDA event timings
 * - Adds runtime dtype selection: fp16 / bf16
 *
 * NEW: runtime init mode selection
 *   --init u01   : uniform random in [0, 1]   (default, original behavior)
 *   --init pm1   : random +/-1 (discrete), like CUTLASS example
 *
 * FP16 mode:
 *   A,B,C are FP16 (cute::half_t)
 *
 * BF16 mode:
 *   A,B are BF16 (cute::bfloat16_t)
 *   GMMA compute/accumulate is FP32 (F32 += BF16*BF16)
 *   C is FP32 output (float)
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstdint>

#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <limits>

#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

#include "check_utils.cuh"

using namespace cute;

// ---------- error checking ----------
static inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
    std::exit(EXIT_FAILURE);
  }
  return result;
}
static inline void checkCurand(curandStatus_t st) {
  if (st != CURAND_STATUS_SUCCESS) {
    std::cerr << "CURAND error: " << int(st) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

// ---------- random fill (device) ----------
template <typename OutT>
__device__ __forceinline__ OutT convert_from_float(float x) { return OutT(x); }
template <>
__device__ __forceinline__ float convert_from_float<float>(float x) { return x; }

template <typename OutT>
__global__ void transformRange(const float* in, OutT* out, float a, float range, size_t n) {
  size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  size_t stride = size_t(gridDim.x) * size_t(blockDim.x);
  for (size_t i = idx; i < n; i += stride) {
    out[i] = convert_from_float<OutT>(in[i] * range + a);
  }
}

template <typename OutT>
__global__ void transformPM1(const float* in, OutT* out, size_t n) {
  size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  size_t stride = size_t(gridDim.x) * size_t(blockDim.x);
  for (size_t i = idx; i < n; i += stride) {
    float v = (in[i] > 0.5f) ? 1.0f : -1.0f;
    out[i] = convert_from_float<OutT>(v);
  }
}

template <typename OutT>
static inline void launch_transform_range(const float* temp, OutT* d_out, size_t n, float a, float b) {
  float range = b - a;
  int threads = 256;
  size_t blocks64 = (n + size_t(threads) - 1) / size_t(threads);
  int blocks = int(std::min<size_t>(blocks64, 65535u));
  transformRange<<<blocks, threads>>>(temp, d_out, a, range, n);
  checkCuda(cudaGetLastError());
}

template <typename OutT>
static inline void launch_transform_pm1(const float* temp, OutT* d_out, size_t n) {
  int threads = 256;
  size_t blocks64 = (n + size_t(threads) - 1) / size_t(threads);
  int blocks = int(std::min<size_t>(blocks64, 65535u));
  transformPM1<<<blocks, threads>>>(temp, d_out, n);
  checkCuda(cudaGetLastError());
}

enum class InitMode { U01, PM1 };

static inline const char* init_mode_str(InitMode m) {
  switch (m) {
    case InitMode::U01: return "u01";
    case InitMode::PM1: return "pm1";
    default:            return "unknown";
  }
}

template <typename OutT>
void GPU_fill_init(OutT* d_out, size_t numElements, unsigned long long seed, InitMode mode) {
  if (numElements == 0) return;

  float* temp = nullptr;
  checkCuda(cudaMalloc(&temp, numElements * sizeof(float)));

  curandGenerator_t prng;
  checkCurand(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
  checkCurand(curandSetPseudoRandomGeneratorSeed(prng, seed));
  checkCurand(curandGenerateUniform(prng, temp, numElements));
  checkCurand(curandDestroyGenerator(prng));

  if (mode == InitMode::PM1) {
    launch_transform_pm1(temp, d_out, numElements);
  } else {
    launch_transform_range(temp, d_out, numElements, 0.0f, 1.0f);
  }

  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(temp));
}

// ---------- Shared storage (from tutorial) ----------
template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

// ---------- MMA atom selection (FP16 vs BF16) ----------
template <typename Element> struct MmaAtomSelectorNT;
template <typename Element> struct MmaAtomSelectorTN;

template <>
struct MmaAtomSelectorNT<cute::half_t> {
  using Atom = SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>;
};
template <>
struct MmaAtomSelectorTN<cute::half_t> {
  using Atom = SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
};

template <>
struct MmaAtomSelectorNT<cute::bfloat16_t> {
  using Atom = SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>;
};
template <>
struct MmaAtomSelectorTN<cute::bfloat16_t> {
  using Atom = SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
};

// ---------- Kernel (from tutorial, structurally unchanged) ----------
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));

  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char shared_memory[];
  using Smem = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  Smem& smem = *reinterpret_cast<Smem*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                   group_modes<0,2>(sA), group_modes<0,2>(gA));

  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                   group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int tma_transaction_bytes =
      sizeof(make_tensor_like(tensor<0>(tAsA))) +
      sizeof(make_tensor_like(tensor<0>(tBsB)));

  auto K_PIPE_MAX = size<1>(tAsA);
  int k_tile_count = size<1>(tAgA);
  int k_tile = 0;

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier;

  CUTE_UNROLL
  for (int pipe = 0; pipe < int(K_PIPE_MAX); ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  cluster_sync();

  CUTE_UNROLL
  for (int pipe = 0; pipe < int(K_PIPE_MAX); ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  auto write_state = cutlass::PipelineState<int(K_PIPE_MAX)>();
  auto read_state  = cutlass::PipelineState<int(K_PIPE_MAX)>();

  CUTE_NO_UNROLL
  while (k_tile_count > -int(K_PIPE_MAX)) {
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate) {
      int pipe = write_state.index();
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  axpby(alpha, tCrC, beta, tCgC);
}

// ---------- GEMM setup (from tutorial) ----------
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k,
             Alpha alpha,
             TA const* A, int ldA,
             TB const* B, int ldB,
             Beta beta,
             TC* C, int ldC)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto bP = Int<  3>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  using Atom = typename MmaAtomSelectorNT<TA>::Atom;
  TiledMMA tiled_mma = make_tiled_mma(Atom{});

  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2,1,1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
    &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                 TA, decltype(sA), decltype(tmaA),
                 TB, decltype(sB), decltype(tmaB),
                 TC, decltype(dC), decltype(tiled_mma),
                 decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
    prob_shape, cta_tiler,
    A, tmaA,
    B, tmaB,
    C, dC, tiled_mma,
    alpha, beta);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: kernel launch failed\n";
  }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k,
             Alpha alpha,
             TA const* A, int ldA,
             TB const* B, int ldB,
             Beta beta,
             TC* C, int ldC)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto bP = Int<  3>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  using Atom = typename MmaAtomSelectorTN<TA>::Atom;
  TiledMMA tiled_mma = make_tiled_mma(Atom{});

  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2,1,1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
    &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                 TA, decltype(sA), decltype(tmaA),
                 TB, decltype(sB), decltype(tmaB),
                 TC, decltype(dC), decltype(tiled_mma),
                 decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
    prob_shape, cta_tiler,
    A, tmaA,
    B, tmaB,
    C, dC, tiled_mma,
    alpha, beta);

  CUTE_CHECK_LAST();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: kernel launch failed\n";
  }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB,
          int m, int n, int k,
          Alpha alpha,
          TA const* A, int ldA,
          TB const* B, int ldB,
          Beta beta,
          TC* C, int ldC)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt<TA,TB,TC>(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC);
  } else if (transA == 'T' && transB == 'N') {
    return gemm_tn<TA,TB,TC>(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC);
  }
  std::cerr << "Unsupported transpose combination. Use NT (default) or TN.\n";
  std::exit(EXIT_FAILURE);
}

// ---------- benchmark harness ----------
struct Args {
  int m = 0, n = 0, k = 0;
  int repeats = 0;
  int warmup = 0;
  int device = 0;
  std::string dtype = "fp16";
  char transA = 'N';
  char transB = 'T';

  // NEW
  std::string init = "u01"; // "u01" or "pm1"

  bool check = false;
  int check_samples = 4096;
  float check_abs = 1e-1f;
  float check_rel = 1e-2f;
  uint64_t check_seed = 1234;
};

static Args parse_args(int argc, char** argv) {
  Args a;

  auto usage = [&](){
    std::cerr <<
    "Usage:\n"
    "  " << argv[0] << " M N K repeats warmup device dtype [transA transB [init]] [options]\n"
    "  " << argv[0] << " --m M --n N --k K --repeats R --warmup W --device D --dtype fp16|bf16 [options]\n\n"
    "Init modes:\n"
    "  init = u01  : uniform random in [0,1]  (default)\n"
    "  init = pm1  : random +/-1 (discrete)\n\n"
    "Options:\n"
    "  --m/--n/--k, --dim, --repeats/--iters, --warmup, --device, --dtype\n"
    "  --transA <N|T>, --transB <N|T>\n"
    "  --init <u01|pm1>\n"
    "  --check --check-samples <int> --check-abs <float> --check-rel <float> --check-seed <int>\n";
    std::exit(EXIT_FAILURE);
  };

  auto is_flag = [&](const char* s){ return s && s[0] == '-'; };

  auto normalize_trans = [&](char x)->char{
    if (x == 't') return 'T';
    if (x == 'n') return 'N';
    return x;
  };

  auto tolower_str = [&](std::string& s){
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });
  };

  auto parse_flags_from = [&](int i){
    for (; i < argc; ++i) {
      std::string key = argv[i];
      auto need = [&](){ if (i + 1 >= argc) usage(); };

      if (key == "--m") { need(); a.m = std::atoi(argv[++i]); }
      else if (key == "--n") { need(); a.n = std::atoi(argv[++i]); }
      else if (key == "--k") { need(); a.k = std::atoi(argv[++i]); }
      else if (key == "--dim") { need(); int dim = std::atoi(argv[++i]); a.m = a.n = a.k = dim; }
      else if (key == "--repeats" || key == "--iters") { need(); a.repeats = std::atoi(argv[++i]); }
      else if (key == "--warmup") { need(); a.warmup = std::atoi(argv[++i]); }
      else if (key == "--device") { need(); a.device = std::atoi(argv[++i]); }
      else if (key == "--dtype") { need(); a.dtype = argv[++i]; }
      else if (key == "--transA") { need(); a.transA = argv[++i][0]; }
      else if (key == "--transB") { need(); a.transB = argv[++i][0]; }
      else if (key == "--init") { need(); a.init = argv[++i]; }
      else if (key == "--check") { a.check = true; }
      else if (key == "--check-samples") { need(); a.check_samples = std::atoi(argv[++i]); }
      else if (key == "--check-abs") { need(); a.check_abs = std::strtof(argv[++i], nullptr); }
      else if (key == "--check-rel") { need(); a.check_rel = std::strtof(argv[++i], nullptr); }
      else if (key == "--check-seed") { need(); a.check_seed = (uint64_t)std::strtoull(argv[++i], nullptr, 10); }
      else if (key == "--help" || key == "-h") { usage(); }
      else { std::cerr << "Unknown arg: " << key << "\n"; usage(); }
    }
  };

  if (argc >= 8 && !is_flag(argv[1])) {
    int i = 1;
    a.m = std::atoi(argv[i++]);
    a.n = std::atoi(argv[i++]);
    a.k = std::atoi(argv[i++]);
    a.repeats = std::atoi(argv[i++]);
    a.warmup  = std::atoi(argv[i++]);
    a.device  = std::atoi(argv[i++]);
    a.dtype   = argv[i++];

    if (i < argc && !is_flag(argv[i])) a.transA = argv[i++][0];
    if (i < argc && !is_flag(argv[i])) a.transB = argv[i++][0];

    // NEW: optional positional init
    if (i < argc && !is_flag(argv[i])) a.init = argv[i++];

    if (i < argc) parse_flags_from(i);
  } else {
    if (argc == 1) usage();
    parse_flags_from(1);
  }

  tolower_str(a.dtype);
  tolower_str(a.init);
  a.transA = normalize_trans(a.transA);
  a.transB = normalize_trans(a.transB);

  if (a.m <= 0 || a.n <= 0 || a.k <= 0) usage();
  if (a.repeats < 0 || a.warmup < 0) usage();
  if (!(a.transA == 'N' || a.transA == 'T')) usage();
  if (!(a.transB == 'N' || a.transB == 'T')) usage();
  if (!(a.dtype == "fp16" || a.dtype == "bf16")) {
    std::cerr << "Unsupported dtype: " << a.dtype << " (expected fp16 or bf16)\n";
    usage();
  }
  if (!(a.init == "u01" || a.init == "pm1")) {
    std::cerr << "Unsupported init: " << a.init << " (expected u01 or pm1)\n";
    usage();
  }

  if ((a.m % 128) != 0 || (a.n % 128) != 0 || (a.k % 64) != 0) {
    std::cerr << "Error: M,N,K must be multiples of (128,128,64). "
              << "Got M=" << a.m << " N=" << a.n << " K=" << a.k << "\n";
    std::exit(EXIT_FAILURE);
  }

  if (a.check) {
    if (a.check_samples <= 0) { std::cerr << "--check-samples must be > 0\n"; usage(); }
    if (a.check_abs < 0.0f || a.check_rel < 0.0f) { std::cerr << "--check-abs/--check-rel must be >= 0\n"; usage(); }
  }

  return a;
}

static inline InitMode parse_init_mode(const Args& a) {
  return (a.init == "pm1") ? InitMode::PM1 : InitMode::U01;
}

// ---------- benchmark runner ----------
template <typename TA, typename TB, typename TC, typename Scalar>
int run_bench(const Args& a)
{
  checkCuda(cudaSetDevice(a.device));
  cudaDeviceProp props{};
  checkCuda(cudaGetDeviceProperties(&props, a.device));
  if (props.major != 9) {
    std::cerr << "This benchmark requires SM90 (Hopper / sm_90a). Detected CC "
              << props.major << "." << props.minor << "\n";
    return 0;
  }

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  std::cerr << "CUTLASS_ARCH_MMA_SM90_SUPPORTED not enabled. Build CUTLASS with SM90 enabled.\n";
  return 0;
#else
  InitMode init_mode = parse_init_mode(a);
  const char* init_str = init_mode_str(init_mode);

  std::cout << "record_type,dtype,tma,init,size,iteration,start_ts,stop_ts,time(ms),verify_pass,verify_rel,verify_abs_floor,ref_acc\n";

  cudaEvent_t init_start, init_stop;
  checkCuda(cudaEventCreate(&init_start));
  checkCuda(cudaEventCreate(&init_stop));

  long long init_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  checkCuda(cudaEventRecord(init_start, 0));

  size_t sizeA = size_t(a.m) * size_t(a.k);
  size_t sizeB = (a.transB == 'T') ? (size_t(a.n) * size_t(a.k)) : (size_t(a.k) * size_t(a.n));
  size_t sizeC = size_t(a.m) * size_t(a.n);

  TA* d_A = nullptr;
  TB* d_B = nullptr;
  TC* d_C = nullptr;
  checkCuda(cudaMalloc(&d_A, sizeA * sizeof(TA)));
  checkCuda(cudaMalloc(&d_B, sizeB * sizeof(TB)));
  checkCuda(cudaMalloc(&d_C, sizeC * sizeof(TC)));

  GPU_fill_init<TA>(d_A, sizeA, 0ULL, init_mode);
  GPU_fill_init<TB>(d_B, sizeB, 1ULL, init_mode);
  checkCuda(cudaMemset(d_C, 0, sizeC * sizeof(TC)));

  checkCuda(cudaEventRecord(init_stop, 0));
  checkCuda(cudaEventSynchronize(init_stop));
  long long init_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  float init_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&init_ms, init_start, init_stop));

  int init_iteration = -a.warmup - 1;
  int size_tag = (a.m == a.n && a.n == a.k) ? a.m : a.m;
  std::cout << "init," << a.dtype << "," << 1 << "," << init_str << "," << size_tag << ","
          << init_iteration << "," << init_start_ts << "," << init_stop_ts << "," << init_ms
          << ",-1,nan,nan," << "" << "\n";


  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  int ldA = (a.transA == 'N') ? a.m : a.k;
  int ldB = (a.transB == 'N') ? a.k : a.n;
  int ldC = a.m;

  Scalar alpha = Scalar(1.0f);
  Scalar beta  = Scalar(0.0f);

  int verify_iteration = -a.warmup - 1;
  int first_rep = a.check ? verify_iteration : -a.warmup;
  bool verify_passed = true;

  for (int rep = first_rep; rep < a.repeats; ++rep) {
    long long iter_start_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    checkCuda(cudaEventRecord(start, 0));

    gemm<TA, TB, TC, Scalar, Scalar>(a.transA, a.transB,
                                     a.m, a.n, a.k,
                                     alpha,
                                     d_A, ldA,
                                     d_B, ldB,
                                     beta,
                                     d_C, ldC);

    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    long long iter_stop_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const char* record_type = (rep < 0 ? "warmup" : "run");
    int verify_pass = -1;
    float verify_rel = std::numeric_limits<float>::quiet_NaN();
    float verify_abs = std::numeric_limits<float>::quiet_NaN();
    const char* ref_acc = "";

    if (a.check && rep == verify_iteration) {
      record_type = "verify";
      checkCuda(cudaDeviceSynchronize());
      bool ok = run_verify<TC, TA, TB>(
          d_C, d_A, d_B,
          a.m, a.n, a.k,
          ldA, ldB, ldC,
          a.transA, a.transB,
          /*abs_tol=*/a.check_abs,
          /*rel_tol=*/a.check_rel,
          /*samples=*/a.check_samples,
          /*seed=*/a.check_seed);
      verify_pass = ok ? 1 : 0;
      verify_rel = a.check_rel;
      verify_abs = a.check_abs;
      ref_acc = "fp32";
      verify_passed = ok;
    }

    std::cout << record_type << "," << a.dtype << "," << 1 << "," << init_str << "," << size_tag << "," << rep << ","
          << iter_start_ts << "," << iter_stop_ts << "," << elapsed_ms << ","
          << verify_pass << "," << verify_rel << "," << verify_abs << "," << ref_acc << "\n";

  }
  if (a.check && !verify_passed) {
    return 1;
  }

  checkCuda(cudaEventDestroy(init_start));
  checkCuda(cudaEventDestroy(init_stop));
  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));

  checkCuda(cudaFree(d_A));
  checkCuda(cudaFree(d_B));
  checkCuda(cudaFree(d_C));
  return 0;
#endif
}

int main(int argc, char** argv)
{
  Args a = parse_args(argc, argv);

  if (a.dtype == "fp16") {
    return run_bench<cute::half_t, cute::half_t,
                     cute::half_t,
                     cute::half_t>(a);
  } else if (a.dtype == "bf16") {
    return run_bench<cute::bfloat16_t, cute::bfloat16_t,
                     float,
                     float>(a);
  }

  std::cerr << "Unsupported dtype: " << a.dtype << " (use fp16 or bf16)\n";
  return 1;
}
