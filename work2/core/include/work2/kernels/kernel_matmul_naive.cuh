#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <work1/entities/kinds.hpp>
#include "../entities/matrix_view.cuh"

namespace hsys::work2::core::kernels {

template <work1::core::AtomK AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> lhs,
                                    MatrixView<AtomT> rhs,
                                    MatrixView<AtomT> result) {

  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  // нить не должна выйти за границы результирующей матрицы
  if (row < result.nrows() && col < result.ncols()) {
    AtomT sum = static_cast<AtomT>(0);
    const std::size_t k_len = lhs.ncols();

    for (std::size_t k = 0; k < k_len; ++k) {
      sum += lhs(row, k) * rhs(k, col);
    }

    result(row, col) = sum;
  }
}

}  // namespace hsys::work2::core::kernels
