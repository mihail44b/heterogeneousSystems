#pragma once

#include "../entities/kinds.hpp"
#include "../entities/vector_view.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace hsys::work1::core::kernels {

template <AtomK AtomT>
__global__ void kernel_vecadd(
    VectorView<AtomT> lhs, VectorView<AtomT> rhs, VectorView<AtomT> result) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < result.size()) {
    result[i] = lhs[i] + rhs[i];
  }
}

}  // namespace hsys::work1::core::kernels
