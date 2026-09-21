#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <hsys/vector.cuh>

#include <vector>

// ═══════════════════════════════════════════════
// Бенчмарк CUDA Vector operator+
// ═══════════════════════════════════════════════

static void BM_CudaVectorAdd(benchmark::State& state) {
  std::size_t n = state.range(0);

  // Создаём векторы ЗАРАНЕЕ, до замера
  // Мы измеряем только скорость сложения, а не аллокации
  std::vector<float> host_data(n, 1.0f);
  hsys::Vector<float> a(host_data);
  hsys::Vector<float> b(host_data);

  // state — объект Google Benchmark
  // Цикл for повторяет операцию много раз для точного замера
  for (auto _ : state) {
    hsys::Vector<float> c = a + b;
    // benchmark::DoNotOptimize не даёт компилятору
    // выбросить результат как "неиспользуемый"
    benchmark::DoNotOptimize(c);
  }
}

// ═══════════════════════════════════════════════
// Бенчмарк Eigen (CPU) — для сравнения
// ═══════════════════════════════════════════════

static void BM_EigenVectorAdd(benchmark::State& state) {
  std::size_t n = state.range(0);

  Eigen::VectorXf a = Eigen::VectorXf::Ones(n);
  Eigen::VectorXf b = Eigen::VectorXf::Ones(n);

  for (auto _ : state) {
    Eigen::VectorXf c = a + b;
    benchmark::DoNotOptimize(c);
  }
}

// Размеры из задания: 8^1, 8^2, ..., 8^8
// RangeMultiplier(8) — каждый следующий размер в 8 раз больше
// Range(8, 8^8) — от 8 до 16'777'216
BENCHMARK(BM_CudaVectorAdd)->RangeMultiplier(8)->Range(8, 1 << 24);
BENCHMARK(BM_EigenVectorAdd)->RangeMultiplier(8)->Range(8, 1 << 24);
