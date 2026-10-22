#pragma once

#include <cuda_fp16.h>
#include <type_traits>

namespace hsys::work1::core {

template <typename T>
concept AtomK = std::is_arithmetic_v<T> || std::is_same_v<T, __half>;

}  // namespace hsys::work1::core
