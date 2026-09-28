#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "../utils/cuda_check.hpp"
#include "kinds.hpp"

namespace hsys::work1::core {

template <AtomK AtomT>
struct Data {
  hsys_data_feature{};

 private:
  std::size_t size_;
  AtomT* data_;

 public:
  using atom_t = AtomT;

  // Создание массива в памяти GPU
  explicit Data(std::size_t size = 1)
      : size_(size)
      , data_(nullptr) {
    if (size_) {
      utils::cuda_check(cudaMalloc(&data_, byte_size()));
    }
  }

  // Copy constructor
  Data(const Data& other)
      : size_(other.size_)
      , data_(nullptr) {
    if (size_) {
      utils::cuda_check(cudaMalloc(&data_, byte_size()));

      utils::cuda_check(
          cudaMemcpy(data_, other.data_, byte_size(), cudaMemcpyDeviceToDevice));
    }
  }

  // Move constructor
  Data(Data&& other) noexcept
      : size_(other.size_)
      , data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;
  }

  // Copy assignment
  Data& operator=(const Data& other) {
    if (this == &other) {
      return *this;
    }

    if (size_ != other.size_) {
      if (data_) {
        cudaFree(data_);
      }

      size_ = other.size_;

      if (size_) {
        utils::cuda_check(cudaMalloc(&data_, byte_size()));
      } else {
        data_ = nullptr;
      }
    }

    if (data_) {
      utils::cuda_check(
          cudaMemcpy(data_, other.data_, byte_size(), cudaMemcpyDeviceToDevice));
    }

    return *this;
  }

  // Move assignment
  Data& operator=(Data&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    if (data_) {
      cudaFree(data_);
    }

    size_ = other.size_;
    data_ = other.data_;

    other.size_ = 0;
    other.data_ = nullptr;

    return *this;
  }

  // Количество элементов
  [[nodiscard]] std::size_t size() const {
    return size_;
  }

  // Размер данных в байтах
  [[nodiscard]] std::size_t byte_size() const {
    return sizeof(AtomT) * size_;
  }

  // Указатель на память GPU
  [[nodiscard]] AtomT* data() {
    return data_;
  }

  [[nodiscard]] const AtomT* data() const {
    return data_;
  }

  // Копирование GPU -> CPU
  void copy_to_host(AtomT* host_ptr) const {
    if (size_ == 0) {
      return;
    }

    utils::cuda_check(cudaMemcpy(host_ptr, data_, byte_size(), cudaMemcpyDeviceToHost));
  }

  // Копирование CPU -> GPU
  void copy_from_host(const AtomT* host_ptr) {
    if (size_ == 0) {
      return;
    }

    utils::cuda_check(cudaMemcpy(data_, host_ptr, byte_size(), cudaMemcpyHostToDevice));
  }

  // Освобождение памяти GPU
  ~Data() {
    if (data_) {
      cudaFree(data_);
    }
  }
};

}  // namespace hsys::work1::core
