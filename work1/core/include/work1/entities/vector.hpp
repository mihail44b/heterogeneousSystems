#pragma once

#include <memory>
#include <vector>

#include "data.hpp"
#include "kinds.hpp"
#include "vector_view.cuh"

namespace hsys::work1::core {

template <AtomK AtomT>
class Vector {
 private:
  std::shared_ptr<Data<AtomT>> data_;
  VectorView<AtomT> view_;

 public:
  using atom_t = AtomT;

  explicit Vector(std::size_t size)
      : data_(std::make_shared<Data<AtomT>>(size))
      , view_(data_->data(), data_->size()) {}

  Vector(const std::vector<AtomT>& host_data)
      : data_(std::make_shared<Data<AtomT>>(host_data.size()))
      , view_(data_->data(), host_data.size()) {
    data_->copy_from_host(host_data.data());
  }

  [[nodiscard]] std::size_t size() const {
    return data_->size();
  }

  [[nodiscard]] Data<AtomT>& data() {
    return *data_;
  }

  [[nodiscard]] const Data<AtomT>& data() const {
    return *data_;
  }

  [[nodiscard]] VectorView<AtomT>& view() {
    return view_;
  }

  [[nodiscard]] const VectorView<AtomT>& view() const {
    return view_;
  }

  [[nodiscard]] std::vector<AtomT> to_host() const {
    std::vector<AtomT> result(size());
    data_->copy_to_host(result.data());
    return result;
  }
};

}  // namespace hsys::work1::core
