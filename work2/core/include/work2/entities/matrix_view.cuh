#pragma once

#include <work1/entities/kinds.hpp>
#include <cstddef>
#include <cuda_runtime.h>

namespace hsys::work2::core {

template <work1::core::AtomK AtomT>
struct MatrixView {
  struct hsys_matrix_view_feature {};

 public:
  using atom_t = AtomT;

 private:
  atom_t* data_;
  std::size_t nrows_;
  std::size_t ncols_;

 public:

  explicit __host__ __device__ MatrixView(atom_t* data, std::size_t nrows, std::size_t ncols)
      : data_(data)
      , nrows_(nrows)
      , ncols_(ncols) {}


  MatrixView(const MatrixView& other) = default;
  MatrixView(MatrixView&& other) = default;
  MatrixView& operator=(const MatrixView& other) = delete;
  MatrixView& operator=(MatrixView&& other) = delete;
  ~MatrixView() = default;


  [[nodiscard]] __host__ __device__ std::size_t nrows() const noexcept {
    return nrows_;
  }

  [[nodiscard]] __host__ __device__ std::size_t ncols() const noexcept {
    return ncols_;
  }

  [[nodiscard]] __host__ __device__ std::size_t size() const noexcept {
    return nrows_ * ncols_;
  }

  [[nodiscard]] __host__ __device__ atom_t* data() noexcept {
    return data_;
  }

  [[nodiscard]] __host__ __device__ const atom_t* data() const noexcept {
    return data_;
  }


  __device__ atom_t& operator[](std::size_t n) {
    return data_[n];
  }

  __device__ const atom_t& operator[](std::size_t n) const {
    return data_[n];
  }

  __device__ atom_t& operator()(std::size_t i, std::size_t j) {
    return data_[i * ncols_ + j];
  }

  __device__ const atom_t& operator()(std::size_t i, std::size_t j) const {
    return data_[i * ncols_ + j];
  }
};

}  // namespace hsys::work2::core
