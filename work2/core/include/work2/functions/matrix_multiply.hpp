#pragma once

#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>

#include <work1/entities/kinds.hpp>
#include <work1/utils/cuda_check.hpp>

#include "../entities/matrix.hpp"
#include "../kernels/kernel_matmul_naive.cuh"

namespace hsys::work2::core {

template <work1::core::AtomK AtomT>
void matrix_multiply(const Matrix<AtomT>& lhs,
                     const Matrix<AtomT>& rhs,
                     Matrix<AtomT>& result) {

  if (lhs.ncols() != rhs.nrows()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible: lhs.ncols() must be equal to rhs.nrows()");
  }

  if (result.nrows() != lhs.nrows() || result.ncols() != rhs.ncols()) {
    throw std::invalid_argument(
        "Result matrix dimensions mismatch: result must be of size lhs.nrows() x rhs.ncols()");
  }


  if (lhs.nrows() == 0 || lhs.ncols() == 0 || rhs.ncols() == 0) {
    return;
  }


  constexpr unsigned int block_dim_x = 16;
  constexpr unsigned int block_dim_y = 16;
  const dim3 block_size(block_dim_x, block_dim_y);


  const dim3 grid_size(
      static_cast<unsigned int>((result.ncols() + block_dim_x - 1) / block_dim_x),
      static_cast<unsigned int>((result.nrows() + block_dim_y - 1) / block_dim_y));


  kernels::kernel_matmul_naive<<<grid_size, block_size>>>(
      lhs.view(), rhs.view(), result.view());


  work1::core::utils::cuda_check(cudaGetLastError());
  work1::core::utils::cuda_check(cudaDeviceSynchronize());
}

template <work1::core::AtomK AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
  Matrix<AtomT> result(lhs.nrows(), rhs.ncols());
  matrix_multiply(lhs, rhs, result);
  return result;
}

}  // namespace hsys::work2::core
