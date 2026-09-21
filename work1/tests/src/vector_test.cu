#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <hsys/vector.cuh>

#include <cmath>
#include <vector>

// Тестируем: CUDA Vector сложение даёт тот же результат, что Eigen на CPU

class VectorAddTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorAddTest, MatchesEigen) {
  std::size_t n = GetParam();

  // 1. Готовим данные на CPU
  std::vector<float> host_a(n);
  std::vector<float> host_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_a[i] = static_cast<float>(i);
    host_b[i] = static_cast<float>(n - i);
  }

  // 2. Считаем на GPU через нашу библиотеку
  hsys::Vector<float> gpu_a(host_a);
  hsys::Vector<float> gpu_b(host_b);
  hsys::Vector<float> gpu_c = gpu_a + gpu_b;
  std::vector<float> gpu_result = gpu_c.to_host();

  // 3. Считаем на CPU через Eigen (эталон)
  Eigen::VectorXf eigen_a = Eigen::Map<Eigen::VectorXf>(host_a.data(), n);
  Eigen::VectorXf eigen_b = Eigen::Map<Eigen::VectorXf>(host_b.data(), n);
  Eigen::VectorXf eigen_c = eigen_a + eigen_b;

  // 4. Сравниваем: CUDA результат ≈ Eigen результат
  Eigen::VectorXf cuda_result = Eigen::Map<Eigen::VectorXf>(gpu_result.data(), n);

  // isApprox проверяет, что разница между векторами < tolerance
  // Преподаватель требует tolerance = 1e-6
  EXPECT_TRUE(cuda_result.isApprox(eigen_c, 1e-6f)) << "Mismatch for size " << n;
}

// Размеры из задания преподавателя:
// 1, 2, 3 — маленькие граничные случаи
// 127, 128, 129 — граница размера CUDA warp/block (128 = степень двойки)
// 512, 1024 — типичные размеры блоков
// 1029 — НЕ степень двойки, проверяет правильность округления в kernel
INSTANTIATE_TEST_SUITE_P(
    Sizes, VectorAddTest, ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029));
