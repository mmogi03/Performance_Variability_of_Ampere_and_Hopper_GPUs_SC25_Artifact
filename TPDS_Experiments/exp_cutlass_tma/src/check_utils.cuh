#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <type_traits>

// CUTLASS core
#include "cutlass/coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/functional.h"

// Using the lower-level reference kernel + comparator utilities.
#include "cutlass/util/reference/device/thread/gemm.h"
#include "cutlass/util/reference/device/kernel/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"

// ------------------------------------------------------------
// Error checking
// ------------------------------------------------------------
static inline void CHECK_CUDA(cudaError_t e, char const* msg = nullptr) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s%s%s\n",
                 cudaGetErrorString(e),
                 msg ? " | " : "",
                 msg ? msg : "");
    std::fflush(stderr);
    std::exit(1);
  }
}

// ------------------------------------------------------------
// Type helpers
// ------------------------------------------------------------
template <typename T>
inline const char* element_str() {
  if (std::is_same<T, float>::value) return "fp32";
  if (std::is_same<T, cutlass::half_t>::value) return "fp16";
  if (std::is_same<T, cutlass::bfloat16_t>::value) return "bf16";
  return "unknown";
}

template <typename To>
inline To scalar_from_float(float x) {
  return cutlass::NumericConverter<To, float>{}(x);
}

// ------------------------------------------------------------
// Output tile descriptor for the reference kernel
// Use enum constants to avoid NVHPC issues with static data members in some contexts.
// ------------------------------------------------------------
struct OutputTile4x4 {
  enum { kRow = 4, kColumn = 4 };
};

// ------------------------------------------------------------
// Device reference GEMM launcher
//
// NOTE: ConvertOp must accept ScalarType (float) because epilogue computes in ScalarType.
// ------------------------------------------------------------
template <typename TensorRefA, typename TensorRefB, typename TensorRefC, typename AccumulatorType>
static inline void reference_device_gemm_launch(
    cutlass::gemm::GemmCoord problem_size,
    float alpha_f,
    TensorRefA tensor_a,
    TensorRefB tensor_b,
    float beta_f,
    TensorRefC tensor_c,
    TensorRefC tensor_d) {

  using ScalarType = float;

  using InnerProductOp = cutlass::multiply_add<AccumulatorType>;

  // FIX: Convert from ScalarType (float) -> output element type
  using ConvertOp = cutlass::NumericConverter<typename TensorRefC::Element, ScalarType>;

  dim3 block(8, 8, 1);
  dim3 grid(
      (problem_size.m() + int(block.x) * OutputTile4x4::kRow - 1) / (int(block.x) * OutputTile4x4::kRow),
      (problem_size.n() + int(block.y) * OutputTile4x4::kColumn - 1) / (int(block.y) * OutputTile4x4::kColumn),
      1);

  cutlass::reference::device::kernel::Gemm<
      TensorRefA,
      TensorRefB,
      TensorRefC,
      ScalarType,
      AccumulatorType,
      OutputTile4x4,
      InnerProductOp,
      ConvertOp
  ><<<grid, block>>>(
      problem_size,
      ScalarType(alpha_f),
      tensor_a,
      tensor_b,
      ScalarType(beta_f),
      tensor_c,
      tensor_d,
      AccumulatorType(0));

  CHECK_CUDA(cudaGetLastError(), "reference device kernel::Gemm launch");
}

// ------------------------------------------------------------
// Verification core
//
// Verifies: D_out ≈ alpha * op(A) * op(B) + beta * C0
//
// Supports NT and TN.
// ------------------------------------------------------------
template <typename TA, typename TB, typename TD>
bool verify_gemm_and_print(
    TD const* d_D_out,
    TA const* d_A,
    TB const* d_B,
    TD const* d_C0,              // may be nullptr if beta == 0
    int m, int n, int k,
    int ldA, int ldB, int ldD, int ldC,
    char transA, char transB,    // 'N' or 'T'
    float alpha_f, float beta_f,
    float rel_tol, float abs_floor, bool do_print = true) {

  const bool isNT = (transA == 'N' || transA == 'n') && (transB == 'T' || transB == 't');
  const bool isTN = (transA == 'T' || transA == 't') && (transB == 'N' || transB == 'n');

  if (!(isNT || isTN)) {
    if (do_print) {
      std::cerr << "[VERIFY] SKIP | unsupported trans=" << transA << transB
                << " (supports NT and TN)\n";
    }
    return false;
  }

  // D and C are stored as (m x n) column-major with leading dimension ld{C,D}
  size_t elemsD = size_t(ldD) * size_t(n);

  // Allocate device reference output
  TD* d_D_ref = nullptr;
  CHECK_CUDA(cudaMalloc(&d_D_ref, elemsD * sizeof(TD)), "cudaMalloc d_D_ref");
  CHECK_CUDA(cudaMemset(d_D_ref, 0, elemsD * sizeof(TD)), "cudaMemset d_D_ref");

  // Prepare C pointer for reference kernel
  TD const* d_C_use = d_C0;
  float beta_use = beta_f;

  TD* d_C_zero = nullptr;
  if (beta_f != 0.0f && d_C0 == nullptr) {
    if (do_print) {
      std::cerr << "[VERIFY] FAIL | beta!=0 but d_C0==nullptr (need original C)\n";
    }
    CHECK_CUDA(cudaFree(d_D_ref));
    return false;
  }
  if (beta_f == 0.0f) {
    // Allocate a zero C so tensor_c is always valid
    size_t elemsC = size_t(ldC) * size_t(n);
    CHECK_CUDA(cudaMalloc(&d_C_zero, elemsC * sizeof(TD)), "cudaMalloc d_C_zero");
    CHECK_CUDA(cudaMemset(d_C_zero, 0, elemsC * sizeof(TD)), "cudaMemset d_C_zero");
    d_C_use = d_C_zero;
    beta_use = 0.0f;
  }

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  using LayoutD = cutlass::layout::ColumnMajor;
  auto Dref = cutlass::TensorRef<TD, LayoutD>(d_D_ref, LayoutD(ldD));
  auto Cref = cutlass::TensorRef<TD, LayoutD>(const_cast<TD*>(d_C_use), LayoutD(ldC));

  // Accumulator choice for reference kernel:
  // For correctness checks, float is safest. If you want to mimic FP16-accum,
  // you can switch this to cutlass::half_t, but float is recommended.
  using AccumulatorType = float;

  if (isNT) {
    // NT:
    // A stored as (m x k) col-major => ColumnMajor view
    // B stored as (n x k) col-major, but op(B)=B^T => interpret as RowMajor (k x n)
    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::RowMajor;

    auto Aref = cutlass::TensorRef<TA, LayoutA>(const_cast<TA*>(d_A), LayoutA(ldA));
    auto Bref = cutlass::TensorRef<TB, LayoutB>(const_cast<TB*>(d_B), LayoutB(ldB));

    reference_device_gemm_launch<decltype(Aref), decltype(Bref), decltype(Cref), AccumulatorType>(
        problem_size, alpha_f, Aref, Bref, beta_use, Cref, Dref);

  } else {
    // TN:
    // A stored as (k x m) col-major, but op(A)=A^T => interpret as RowMajor (m x k)
    // B stored as (k x n) col-major => ColumnMajor view
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;

    auto Aref = cutlass::TensorRef<TA, LayoutA>(const_cast<TA*>(d_A), LayoutA(ldA));
    auto Bref = cutlass::TensorRef<TB, LayoutB>(const_cast<TB*>(d_B), LayoutB(ldB));

    reference_device_gemm_launch<decltype(Aref), decltype(Bref), decltype(Cref), AccumulatorType>(
        problem_size, alpha_f, Aref, Bref, beta_use, Cref, Dref);
  }

  CHECK_CUDA(cudaDeviceSynchronize(), "sync reference device gemm");

  // Device-side compare
  bool passed = false;
  {
    auto const* ref_ptr = reinterpret_cast<TD const*>(d_D_ref);
    auto const* out_ptr = reinterpret_cast<TD const*>(d_D_out);

    TD eps   = scalar_from_float<TD>(rel_tol);
    TD floor = scalar_from_float<TD>(abs_floor);

    passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        ref_ptr, out_ptr, elemsD, eps, floor);
  }

  if (do_print) {
    std::cout
      << "[VERIFY] " << (passed ? "PASS" : "FAIL")
      << " | A=" << element_str<TA>()
      << " B=" << element_str<TB>()
      << " out=" << element_str<TD>()
      << " | trans=" << (isNT ? "NT" : "TN")
      << " | m=" << m << " n=" << n << " k=" << k
      << " | rel=" << rel_tol << " abs_floor=" << abs_floor
      << " | ref_acc=fp32"
      << std::endl;
  }

  if (d_C_zero) CHECK_CUDA(cudaFree(d_C_zero), "cudaFree d_C_zero");
  CHECK_CUDA(cudaFree(d_D_ref), "cudaFree d_D_ref");

  return passed;
}

// ------------------------------------------------------------
// Benchmark-facing wrapper
// Assumes alpha=1, beta=0.
// ------------------------------------------------------------
template <typename TC, typename TA, typename TB>
bool run_verify(
    TC* d_C_out,
    TA* d_A,
    TB* d_B,
    int m, int n, int k,
    int ldA, int ldB, int ldC,
    char transA, char transB,
    float abs_tol, float rel_tol,
    int /*samples*/ = 0,
    uint64_t /*seed*/ = 0) {

  return verify_gemm_and_print<TA, TB, TC>(
      d_C_out, d_A, d_B,
      /*d_C0=*/nullptr,
      m, n, k,
      ldA, ldB,
      /*ldD=*/ldC,
      /*ldC=*/ldC,
      transA, transB,
      /*alpha=*/1.0f,
      /*beta=*/0.0f,
      /*rel_tol=*/rel_tol,
      /*abs_floor=*/abs_tol,
      /*do_print=*/false);
}

template <typename TC, typename TA, typename TB>
void run_verify_and_print(
    TC* d_C_out,
    TA* d_A,
    TB* d_B,
    int m, int n, int k,
    int ldA, int ldB, int ldC,
    char transA, char transB,
    float abs_tol, float rel_tol,
    int /*samples*/ = 0,
    uint64_t /*seed*/ = 0) {

  (void)verify_gemm_and_print<TA, TB, TC>(
      d_C_out, d_A, d_B,
      /*d_C0=*/nullptr,
      m, n, k,
      ldA, ldB,
      /*ldD=*/ldC,
      /*ldC=*/ldC,
      transA, transB,
      /*alpha=*/1.0f,
      /*beta=*/0.0f,
      /*rel_tol=*/rel_tol,
      /*abs_floor=*/abs_tol,
      /*do_print=*/true);
}
