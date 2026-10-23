#pragma once

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

#include <work1/entities/kinds.hpp>
#include <work2/entities/matrix_view.cuh>

namespace hsys::work4::core::kernels {

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <work1::core::AtomK AtomT>
__global__ void kernel_matmul_wmma(work2::core::MatrixView<AtomT> lhs,
    work2::core::MatrixView<AtomT> rhs,
    work2::core::MatrixView<AtomT> result) {
  const std::size_t warp_x
      = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warpSize;
  const std::size_t warp_y
      = (static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y);

  const std::size_t i = warp_y * WMMA_M;
  const std::size_t j = warp_x * WMMA_N;

  if (i >= result.nrows() || j >= result.ncols()) {
    return;
  }

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  wmma::fill_fragment(c_frag, __float2half(0.0f));

  const std::size_t k_len = lhs.ncols();

  for (std::size_t v = 0; v < k_len; v += WMMA_K) {
    wmma::load_matrix_sync(a_frag, &lhs(i, v), lhs.ncols());
    wmma::load_matrix_sync(b_frag, &rhs(v, j), rhs.ncols());
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(&result(i, j), c_frag, result.ncols(), wmma::mem_row_major);
}

}  // namespace hsys::work4::core::kernels
