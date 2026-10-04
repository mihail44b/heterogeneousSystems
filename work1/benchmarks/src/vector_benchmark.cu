#include <new>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <work1/hsys.cuh>

#include <vector>

// ============================================================
// Бенчмарк CUDA Vector (замер чистого времени ядра через CUDA Events API)
// ============================================================

static void BM_CudaVectorAdd(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  // Выделяем память и заполняем данные ДО замера времени
  std::vector<float> host_a(n, 1.0f);
  std::vector<float> host_b(n, 2.0f);

  hsys::Vector<float> a(host_a);
  hsys::Vector<float> b(host_b);
  hsys::Vector<float> c(n);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  hsys::work1::core::utils::cuda_check(cudaEventCreate(&start));
  hsys::work1::core::utils::cuda_check(cudaEventCreate(&stop));

  for (auto _ : state) {
    hsys::work1::core::utils::cuda_check(cudaEventRecord(start));

    // Измеряем ТОЛЬКО сложение без аллокаций памяти
    hsys::vector_add(a, b, c);

    hsys::work1::core::utils::cuda_check(cudaEventRecord(stop));
    hsys::work1::core::utils::cuda_check(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    hsys::work1::core::utils::cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));

    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
  }

  hsys::work1::core::utils::cuda_check(cudaEventDestroy(start));
  hsys::work1::core::utils::cuda_check(cudaEventDestroy(stop));

  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// ============================================================
// Бенчмарк Eigen (CPU) — эталон для сравнения
// ============================================================

static void BM_EigenVectorAdd(benchmark::State& state) {
  const Eigen::Index n = static_cast<Eigen::Index>(state.range(0));

  Eigen::VectorXf a = Eigen::VectorXf::Ones(n);
  Eigen::VectorXf b = Eigen::VectorXf::Constant(n, 2.0f);
  Eigen::VectorXf c(n);

  for (auto _ : state) {
    c.noalias() = a + b;
    benchmark::DoNotOptimize(c.data());
  }

  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// Размеры из методички: 8^1, 8^2, ..., 8^8 (от 8 до 16'777'216)
BENCHMARK(BM_CudaVectorAdd)
    ->RangeMultiplier(8)
    ->Range(8, 1 << 24)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_EigenVectorAdd)
    ->RangeMultiplier(8)
    ->Range(8, 1 << 24)
    ->Unit(benchmark::kMicrosecond);
