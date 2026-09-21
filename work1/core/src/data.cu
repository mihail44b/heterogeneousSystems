#include <hsys/data.cuh>

#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace hsys {

template <typename T>
Data<T>::Data(std::size_t size)
    : size_(size) {
  cudaError_t err = cudaMalloc(&ptr_, size * sizeof(T));
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

template <typename T>
Data<T>::~Data() {
  if (ptr_) {
    cudaFree(ptr_);
  }
}

template <typename T>
Data<T>::Data(const Data& other)
    : size_(other.size_) {
  cudaMalloc(&ptr_, size_ * sizeof(T));
  cudaMemcpy(ptr_, other.ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T>
Data<T>::Data(Data&& other) noexcept
    : ptr_(other.ptr_)
    , size_(other.size_) {
  other.ptr_ = nullptr;
  other.size_ = 0;
}

template <typename T>
Data<T>& Data<T>::operator=(const Data& other) {
  if (this != &other) {
    if (ptr_) {
      cudaFree(ptr_);
    }
    size_ = other.size_;
    cudaMalloc(&ptr_, size_ * sizeof(T));
    cudaMemcpy(ptr_, other.ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  return *this;
}

template <typename T>
Data<T>& Data<T>::operator=(Data&& other) noexcept {
  if (this != &other) {
    if (ptr_) {
      cudaFree(ptr_);
    }
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

template <typename T>
void Data<T>::copy_from_host(const T* host_data, std::size_t count) {
  cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void Data<T>::copy_to_host(T* host_data, std::size_t count) const {
  cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
}

template class Data<float>;

}  // namespace hsys
