#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <work1/entities/kinds.hpp>
#include <work2/entities/matrix_view.cuh>

namespace hsys::work3::core::kernels {

constexpr unsigned int TILE_SIZE = 16;

template <work1::core::AtomK AtomT>
__global__ void kernel_matmul_shmem(work2::core::MatrixView<AtomT> lhs,
    work2::core::MatrixView<AtomT> rhs,
    work2::core::MatrixView<AtomT> result) {

  __shared__ AtomT s_a[TILE_SIZE][TILE_SIZE];
  __shared__ AtomT s_b[TILE_SIZE][TILE_SIZE];

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * TILE_SIZE + tx;
  const std::size_t row = static_cast<std::size_t>(blockIdx.y) * TILE_SIZE + ty;

  AtomT sum = static_cast<AtomT>(0);

  const std::size_t k_len = lhs.ncols();
  const std::size_t num_tiles = (k_len + TILE_SIZE - 1) / TILE_SIZE;

  for (std::size_t t = 0; t < num_tiles; ++t) {
    const std::size_t a_col = t * TILE_SIZE + tx;
    if (row < lhs.nrows() && a_col < lhs.ncols()) {
      s_a[ty][tx] = lhs(row, a_col);
    } else {
      s_a[ty][tx] = static_cast<AtomT>(0);
    }

    const std::size_t b_row = t * TILE_SIZE + ty;
    if (b_row < rhs.nrows() && col < rhs.ncols()) {
      s_b[ty][tx] = rhs(b_row, col);
    } else {
      s_b[ty][tx] = static_cast<AtomT>(0);
    }

    __syncthreads();

    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
      sum += s_a[ty][k] * s_b[k][tx];
    }

    __syncthreads();
  }

  if (row < result.nrows() && col < result.ncols()) {
    result(row, col) = sum;
  }
}

}  // namespace hsys::work3::core::kernels
