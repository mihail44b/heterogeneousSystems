#include <benchmark/benchmark.h>

#include <Eigen/Dense>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "functions/vector_add.hpp"
#include "utils/cuda_check.hpp"
#include "vector.hpp"

using hsys::work1::core::utils::cuda_check;
using hsys::work1::core::Vector;
using hsys::work1::core::vector_add;

// ============================================================
// CUDA Vector Add
// ============================================================

static void BM_CudaVectorAdd(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  // ----------------------------------------------------------
  // Подготовка.
  //
  // Всё это происходит ДО benchmark loop:
  //   - host allocation
  //   - device allocation
  //   - Host -> Device copy
  // ----------------------------------------------------------

  std::vector<float> host_a(n, 1.0f);
  std::vector<float> host_b(n, 2.0f);

  Vector<float> a(n);
  Vector<float> b(n);
  Vector<float> c(n);

  a.data().copy_from_host(host_a.data());
  b.data().copy_from_host(host_b.data());

  // ----------------------------------------------------------
  // CUDA Events
  // ----------------------------------------------------------

  cudaEvent_t start{};
  cudaEvent_t stop{};

  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&stop));

  for (auto _ : state) {

    cuda_check(cudaEventRecord(start));

    // --------------------------------------------------------
    // Измеряем ТОЛЬКО vector addition.
    //
    // Здесь НЕТ cudaMalloc/cudaFree.
    // --------------------------------------------------------

    vector_add(a, b, c);

    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;

    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Google Benchmark получает время одной итерации
    // в секундах.
    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
  }

  cuda_check(cudaEventDestroy(start));
  cuda_check(cudaEventDestroy(stop));

  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// ============================================================
// Eigen Vector Add
// ============================================================

static void BM_EigenVectorAdd(benchmark::State& state) {
  const Eigen::Index n = static_cast<Eigen::Index>(state.range(0));

  // ----------------------------------------------------------
  // Подготовка ДО benchmark loop.
  // ----------------------------------------------------------

  Eigen::VectorXf a = Eigen::VectorXf::Ones(n);

  Eigen::VectorXf b = Eigen::VectorXf::Constant(n, 2.0f);

  Eigen::VectorXf c(n);

  // ----------------------------------------------------------
  // Измерение Eigen.
  // ----------------------------------------------------------

  for (auto _ : state) {
    c.noalias() = a + b;

    benchmark::DoNotOptimize(c.data());
  }

  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// ============================================================
// Registration
// ============================================================

BENCHMARK(BM_CudaVectorAdd)->RangeMultiplier(2)->Range(8, 1 << 20)->UseManualTime();

BENCHMARK(BM_EigenVectorAdd)->RangeMultiplier(2)->Range(8, 1 << 20);

BENCHMARK_MAIN();
