#pragma once

#include "../entities/kinds.hpp"
#include "../entities/vector_view.cuh"

namespace hsys::work1::core::kernels {

template <AtomK AtomT>
__global__ void vector_add(
    VectorView<AtomT> lhs, VectorView<AtomT> rhs, VectorView<AtomT> result) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < result.size()) {
    result[i] = lhs[i] + rhs[i];
  }
}

template <AtomK AtomT>
__global__ void kernel_vecadd(
    VectorView<AtomT> lhs, VectorView<AtomT> rhs, VectorView<AtomT> result) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < result.size()) {
    result[i] = lhs[i] + rhs[i];
  }
}

}  // namespace hsys::work1::core::kernels
