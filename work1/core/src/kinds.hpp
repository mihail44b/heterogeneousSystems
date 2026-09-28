#pragma once

#include <type_traits>

namespace hsys::work1::core {

template <typename T>
concept AtomK = std::is_arithmetic_v<T>;

}  // namespace hsys::work1::core
