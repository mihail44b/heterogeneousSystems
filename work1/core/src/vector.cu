#include <hsys/vector.cuh>

#include <cuda_runtime.h>
#include <stdexcept>

namespace hsys {

// ═══════════════════════════════════════════════
// CUDA KERNEL — этот код выполняется на GPU
// ═══════════════════════════════════════════════

// __global__ означает: эта функция запускается с CPU,
// но выполняется на GPU тысячами потоков параллельно
template <typename T>
__global__ void kernel_vecadd(
    VectorView<T> a,  // передаётся по значению (trivially copyable!)
    VectorView<T> b,
    VectorView<T> result) {
  // Каждый поток GPU получает свой уникальный индекс:
  //   blockIdx.x  — номер блока (группы потоков)
  //   blockDim.x  — сколько потоков в блоке
  //   threadIdx.x — номер потока внутри блока
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // Потоков может быть больше, чем элементов (округление вверх),
  // поэтому проверяем границу
  if (i < a.size()) {
    result[i] = a[i] + b[i];
  }
}

// ═══════════════════════════════════════════════
// Vector — реализация методов
// ═══════════════════════════════════════════════

template <typename T>
Vector<T>::Vector(std::size_t size)
    : data_(std::make_shared<Data<T>>(size))
    , view_(data_->ptr(), size) {}

template <typename T>
Vector<T>::Vector(const std::vector<T>& host_data)
    : data_(std::make_shared<Data<T>>(host_data.size()))
    , view_(data_->ptr(), host_data.size()) {
  // Копируем данные из обычного std::vector (CPU) в GPU-память
  data_->copy_from_host(host_data.data(), host_data.size());
}

template <typename T>
std::vector<T> Vector<T>::to_host() const {
  // Создаём обычный std::vector на CPU и копируем данные с GPU
  std::vector<T> result(size());
  data_->copy_to_host(result.data(), size());
  return result;
}

// ═══════════════════════════════════════════════
// operator+ — здесь запускается GPU kernel
// ═══════════════════════════════════════════════

template <typename T>
Vector<T> operator+(const Vector<T>& a, const Vector<T>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("Vector sizes do not match");
  }

  Vector<T> result(a.size());

  // Конфигурация запуска kernel:
  //   threads_per_block — сколько потоков в одном блоке GPU
  //   num_blocks — сколько блоков нужно, чтобы покрыть все элементы
  //
  // Например, для 1000 элементов и 256 потоков на блок:
  //   num_blocks = (1000 + 255) / 256 = 4 блока
  //   Всего потоков = 4 * 256 = 1024 (24 лишних, но kernel проверяет if)
  constexpr int threads_per_block = 256;
  int num_blocks = (a.size() + threads_per_block - 1) / threads_per_block;

  // <<<num_blocks, threads_per_block>>> — синтаксис запуска CUDA kernel
  // Это НЕ обычный C++. Это расширение NVCC.
  // Означает: "запусти kernel_vecadd на GPU с такой конфигурацией потоков"
  kernel_vecadd<<<num_blocks, threads_per_block>>>(a.view_, b.view_, result.view_);

  // cudaDeviceSynchronize() — ждём, пока GPU закончит вычисления
  // Без этого CPU побежит дальше, а GPU ещё считает
  cudaDeviceSynchronize();

  return result;
}

// Явная инстанциация для float
template class Vector<float>;
template Vector<float> operator+(const Vector<float>&, const Vector<float>&);

}  // namespace hsys
