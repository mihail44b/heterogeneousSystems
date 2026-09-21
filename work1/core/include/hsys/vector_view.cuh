#pragma once

#include <cstddef>

namespace hsys {

template <typename T>
class VectorView {
 public:
  VectorView() = default;

  VectorView(T* data, std::size_t size)
      : data_(data)
      , size_(size) {}

  __host__ __device__ T& operator[](std::size_t i) {
    return data_[i];
  }

  __host__ __device__ const T& operator[](std::size_t i) const {
    return data_[i];
  }

  __host__ __device__ T& operator()(std::size_t i) {
    return data_[i];
  }

  __host__ __device__ const T& operator()(std::size_t i) const {
    return data_[i];
  }

  __host__ __device__ std::size_t size() const {
    return size_;
  }

  T* data() {
    return data_;
  }

  const T* data() const {
    return data_;
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace hsys
