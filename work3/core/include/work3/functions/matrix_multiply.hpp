#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>

#include <work1/entities/kinds.hpp>
#include <work1/utils/cuda_check.hpp>
#include <work2/entities/matrix.hpp>
#include <work2/kernels/kernel_matmul_naive.cuh>

#include "../kernels/kernel_matmul_shmem.cuh"

namespace hsys::work3::core {

using work2::core::Matrix;

enum class Strategy { Naive, SharedMemory };

template <work1::core::AtomK AtomT>
void matrix_multiply_naive(
    const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs, Matrix<AtomT>& result) {

  if (lhs.ncols() != rhs.nrows()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible: lhs.ncols() must be equal to rhs.nrows()");
  }

  if (result.nrows() != lhs.nrows() || result.ncols() != rhs.ncols()) {
    throw std::invalid_argument(
        "Result matrix dimensions mismatch: result must be of size lhs.nrows() x "
        "rhs.ncols()");
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

  work2::core::kernels::kernel_matmul_naive<<<grid_size, block_size>>>(
      lhs.view(), rhs.view(), result.view());

  work1::core::utils::cuda_check(cudaGetLastError());
  work1::core::utils::cuda_check(cudaDeviceSynchronize());
}

template <work1::core::AtomK AtomT>
void matrix_multiply_shmem(
    const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs, Matrix<AtomT>& result) {

  if (lhs.ncols() != rhs.nrows()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible: lhs.ncols() must be equal to rhs.nrows()");
  }

  if (result.nrows() != lhs.nrows() || result.ncols() != rhs.ncols()) {
    throw std::invalid_argument(
        "Result matrix dimensions mismatch: result must be of size lhs.nrows() x "
        "rhs.ncols()");
  }

  if (lhs.nrows() == 0 || lhs.ncols() == 0 || rhs.ncols() == 0) {
    return;
  }

  const dim3 block_size(kernels::TILE_SIZE, kernels::TILE_SIZE);
  const dim3 grid_size(static_cast<unsigned int>((result.ncols() + kernels::TILE_SIZE - 1)
                                                 / kernels::TILE_SIZE),
      static_cast<unsigned int>(
          (result.nrows() + kernels::TILE_SIZE - 1) / kernels::TILE_SIZE));

  kernels::kernel_matmul_shmem<<<grid_size, block_size>>>(
      lhs.view(), rhs.view(), result.view());

  work1::core::utils::cuda_check(cudaGetLastError());
  work1::core::utils::cuda_check(cudaDeviceSynchronize());
}

template <work1::core::AtomK AtomT>
void matrix_multiply(const Matrix<AtomT>& lhs,
    const Matrix<AtomT>& rhs,
    Matrix<AtomT>& result,
    Strategy strategy = Strategy::SharedMemory) {
  if (strategy == Strategy::Naive) {
    matrix_multiply_naive(lhs, rhs, result);
  } else {
    matrix_multiply_shmem(lhs, rhs, result);
  }
}

template <work1::core::AtomK AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
  Matrix<AtomT> result(lhs.nrows(), rhs.ncols());
  matrix_multiply(lhs, rhs, result, Strategy::SharedMemory);
  return result;
}

}  // namespace hsys::work3::core
