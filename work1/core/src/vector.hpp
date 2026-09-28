#pragma once

#include <memory>

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
};

}  // namespace hsys::work1::core
