#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace hsys {

template <typename T>
class Data {
 public:
  Data() = default;
  explicit Data(std::size_t size);
  ~Data();

  Data(const Data& other);
  Data(Data&& other) noexcept;
  Data& operator=(const Data& other);
  Data& operator=(Data&& other) noexcept;

  void copy_from_host(const T* host_data, std::size_t count);
  void copy_to_host(T* host_data, std::size_t count) const;

  T* ptr() noexcept {
    return ptr_;
  }

  const T* ptr() const noexcept {
    return ptr_;
  }

  std::size_t size() const noexcept {
    return size_;
  }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace hsys
