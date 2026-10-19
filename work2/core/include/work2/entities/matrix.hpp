#pragma once

#include <memory>
#include <vector>

#include <work1/entities/data.hpp>
#include <work1/entities/kinds.hpp>
#include "matrix_view.cuh"

namespace hsys::work2::core {

template <work1::core::AtomK AtomT>
class Matrix {
 private:
  std::shared_ptr<work1::core::Data<AtomT>> data_;
  MatrixView<AtomT> view_;

 public:
  using atom_t = AtomT;

  Matrix(std::size_t nrows, std::size_t ncols)
      : data_(std::make_shared<work1::core::Data<AtomT>>(nrows * ncols))
      , view_(data_->data(), nrows, ncols) {}

  Matrix(std::size_t nrows, std::size_t ncols, const std::vector<AtomT>& host_data)
      : data_(std::make_shared<work1::core::Data<AtomT>>(nrows * ncols))
      , view_(data_->data(), nrows, ncols) {
    data_->copy_from_host(host_data.data());
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return view_.size();
  }

  [[nodiscard]] std::size_t nrows() const noexcept {
    return view_.nrows();
  }

  [[nodiscard]] std::size_t ncols() const noexcept {
    return view_.ncols();
  }

  [[nodiscard]] work1::core::Data<AtomT>& data() noexcept {
    return *data_;
  }

  [[nodiscard]] const work1::core::Data<AtomT>& data() const noexcept {
    return *data_;
  }

  [[nodiscard]] MatrixView<AtomT>& view() noexcept {
    return view_;
  }

  [[nodiscard]] const MatrixView<AtomT>& view() const noexcept {
    return view_;
  }

  [[nodiscard]] std::vector<AtomT> to_host() const {
    std::vector<AtomT> result(size());
    data_->copy_to_host(result.data());
    return result;
  }
};

}  // namespace hsys::work2::core
