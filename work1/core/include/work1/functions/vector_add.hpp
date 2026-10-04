#pragma once

#include "../entities/kinds.hpp"
#include "../entities/vector.hpp"
#include "../kernels/vector_add.cuh"
#include "../utils/cuda_check.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace hsys::work1::core {

template <AtomK AtomT>
void vector_add(
    const Vector<AtomT>& lhs, const Vector<AtomT>& rhs, Vector<AtomT>& result) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("Vector sizes must be equal");
  }

  if (result.size() != lhs.size()) {
    throw std::invalid_argument("Result vector must have the same size");
  }

  if (lhs.size() == 0) {
    return;
  }

  constexpr unsigned int block_size = 256;
  const unsigned int grid_size =
      static_cast<unsigned int>((lhs.size() + block_size - 1) / block_size);

  kernels::vector_add<<<grid_size, block_size>>>(lhs.view(), rhs.view(), result.view());

  utils::cuda_check(cudaGetLastError());
  utils::cuda_check(cudaDeviceSynchronize());
}

template <AtomK AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs) {
  Vector<AtomT> result(lhs.size());
  vector_add(lhs, rhs, result);
  return result;
}

}  // namespace hsys::work1::core
