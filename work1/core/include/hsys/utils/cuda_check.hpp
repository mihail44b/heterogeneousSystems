#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hsys::work1::core::utils {

inline void cuda_check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
  }
}

}  // namespace hsys::work1::core::utils
