#pragma once

#include "kinds.hpp"

namespace hsys::work1::core {

template <AtomK AtomT>
struct VectorView {
  struct hsys_vector_view_feature {};

 public:
  using atom_t = AtomT;

 private:
  atom_t* data_;
  std::size_t size_;

 public:
  explicit __host__ __device__ VectorView(atom_t* data, std::size_t size)
      : data_(data)
      , size_(size) {}

  VectorView(const VectorView& other) = default;
  VectorView(VectorView&& other) = default;

  VectorView& operator=(const VectorView& other) = delete;
  VectorView& operator=(VectorView&& other) = delete;

  ~VectorView() = default;

  [[nodiscard]] __host__ __device__ std::size_t size() const {
    return size_;
  }

  [[nodiscard]] __host__ __device__ atom_t* data() noexcept {
    return data_;
  }

  [[nodiscard]] __host__ __device__ const atom_t* data() const noexcept {
    return data_;
  }

  __device__ atom_t& operator[](std::size_t i) {
    return data_[i];
  }

  __device__ const atom_t& operator[](std::size_t i) const {
    return data_[i];
  }

  __device__ atom_t& operator()(std::size_t i) {
    return data_[i];
  }

  __device__ const atom_t& operator()(std::size_t i) const {
    return data_[i];
  }
};

}  // namespace hsys::work1::core
