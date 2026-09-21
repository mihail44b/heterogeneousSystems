#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <hsys/data.cuh>
#include <hsys/vector_view.cuh>

namespace hsys {

template <typename T>
class Vector {
 public:
  Vector() = default;
  explicit Vector(std::size_t size);

  Vector(const std::vector<T>& host_data);

  std::vector<T> to_host() const;

  VectorView<T> view() {
    return view_;
  }

  VectorView<const T> view() const {
    return {data_->ptr(), data_->size()};
  }

  std::size_t size() const {
    return data_ ? data_->size() : 0;
  }

  template <typename U>
  friend Vector<U> operator+(const Vector<U>& a, const Vector<U>& b);

 private:
  std::shared_ptr<Data<T>> data_;
  VectorView<T> view_;
};

}  // namespace hsys
