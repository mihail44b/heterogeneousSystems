#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <vector>

#include <work1/utils/cuda_check.hpp>
#include <work2/entities/matrix.hpp>
#include <work2/functions/matrix_multiply.hpp>

using hsys::work2::core::Matrix;
using hsys::work2::core::matrix_multiply;
using hsys::work1::core::utils::cuda_check;

// бенчмарк умножения матриц на GPU (CUDA Events API)
static void BM_CudaMatrixMul(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));


  std::vector<float> host_a(n * n, 1.0f);
  std::vector<float> host_b(n * n, 2.0f);

  Matrix<float> a(n, n, host_a);
  Matrix<float> b(n, n, host_b);
  Matrix<float> c(n, n);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&stop));

  for (auto _ : state) {
    cuda_check(cudaEventRecord(start));


    matrix_multiply(a, b, c);   // вычисление

    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));


    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
  }

  cuda_check(cudaEventDestroy(start));
  cuda_check(cudaEventDestroy(stop));

  // число flop: 2 * N^3
  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * 2 * static_cast<int64_t>(n) * static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

// бенчмарк умножения матриц на CPU (Eigen)
static void BM_EigenMatrixMul(benchmark::State& state) {
  const Eigen::Index n = static_cast<Eigen::Index>(state.range(0));

  Eigen::MatrixXf a = Eigen::MatrixXf::Ones(n, n);
  Eigen::MatrixXf b = Eigen::MatrixXf::Constant(n, n, 2.0f);
  Eigen::MatrixXf c(n, n);

  for (auto _ : state) {
    c.noalias() = a * b;
    benchmark::DoNotOptimize(c.data());
  }

  state.SetItemsProcessed(
      static_cast<int64_t>(state.iterations()) * 2 * static_cast<int64_t>(n) * static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

// диапазон {16, 32, 64, 128, 256, 512, 1024}
BENCHMARK(BM_CudaMatrixMul)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_EigenMatrixMul)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->Unit(benchmark::kMicrosecond);
