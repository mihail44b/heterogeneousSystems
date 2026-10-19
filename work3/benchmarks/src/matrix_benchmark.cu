#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include <work1/utils/cuda_check.hpp>
#include <work2/entities/matrix.hpp>
#include <work3/functions/matrix_multiply.hpp>

using hsys::work1::core::utils::cuda_check;
using hsys::work2::core::Matrix;
using hsys::work3::core::matrix_multiply;
using hsys::work3::core::Strategy;

template <Strategy S>
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

    matrix_multiply(a, b, c, S);

    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));

    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
  }

  cuda_check(cudaEventDestroy(start));
  cuda_check(cudaEventDestroy(stop));

  // 2 * N^3 FLOPs for NxN matrix multiplication
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2
                          * static_cast<int64_t>(n) * static_cast<int64_t>(n)
                          * static_cast<int64_t>(n));
}

static void BM_CudaMatrixMulNaive(benchmark::State& state) {
  BM_CudaMatrixMul<Strategy::Naive>(state);
}

static void BM_CudaMatrixMulShmem(benchmark::State& state) {
  BM_CudaMatrixMul<Strategy::SharedMemory>(state);
}

BENCHMARK(BM_CudaMatrixMulNaive)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_CudaMatrixMulShmem)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
