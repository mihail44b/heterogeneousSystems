#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <work1/hsys.cuh>

#include <cmath>
#include <vector>

// Тестируем: сложение векторов на GPU даёт тот же результат, что Eigen на CPU

class VectorAddTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorAddTest, MatchesEigenOperatorPlus) {
  const std::size_t n = GetParam();

  // 1. Готовим данные на CPU
  std::vector<float> host_a(n);
  std::vector<float> host_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_a[i] = static_cast<float>(i);
    host_b[i] = static_cast<float>(n - i);
  }

  // 2. Считаем на GPU через operator+
  hsys::Vector<float> gpu_a(host_a);
  hsys::Vector<float> gpu_b(host_b);
  hsys::Vector<float> gpu_c = gpu_a + gpu_b;
  std::vector<float> gpu_result = gpu_c.to_host();

  // 3. Считаем на CPU через Eigen (эталон)
  Eigen::VectorXf eigen_a = Eigen::Map<Eigen::VectorXf>(host_a.data(), n);
  Eigen::VectorXf eigen_b = Eigen::Map<Eigen::VectorXf>(host_b.data(), n);
  Eigen::VectorXf eigen_c = eigen_a + eigen_b;

  // 4. Сравниваем: CUDA результат ≈ Eigen результат (точность 1e-6)
  Eigen::VectorXf cuda_result = Eigen::Map<Eigen::VectorXf>(gpu_result.data(), n);
  EXPECT_TRUE(cuda_result.isApprox(eigen_c, 1e-6f)) << "Mismatch for size " << n;
}

TEST_P(VectorAddTest, MatchesEigenFunction) {
  const std::size_t n = GetParam();

  std::vector<float> host_a(n);
  std::vector<float> host_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_a[i] = static_cast<float>(i * 2);
    host_b[i] = static_cast<float>(n + i);
  }

  hsys::Vector<float> gpu_a(host_a);
  hsys::Vector<float> gpu_b(host_b);
  hsys::Vector<float> gpu_c(n);

  // Считаем через явную функцию vector_add
  hsys::vector_add(gpu_a, gpu_b, gpu_c);
  std::vector<float> gpu_result = gpu_c.to_host();

  Eigen::VectorXf eigen_a = Eigen::Map<Eigen::VectorXf>(host_a.data(), n);
  Eigen::VectorXf eigen_b = Eigen::Map<Eigen::VectorXf>(host_b.data(), n);
  Eigen::VectorXf eigen_c = eigen_a + eigen_b;

  Eigen::VectorXf cuda_result = Eigen::Map<Eigen::VectorXf>(gpu_result.data(), n);
  EXPECT_TRUE(cuda_result.isApprox(eigen_c, 1e-6f)) << "Mismatch for size " << n;
}

// Размеры из методички:
// 1, 2, 3 — краевые случаи
// 127, 128, 129 — границы варпов и блоков
// 512, 1024 — типичные размеры блоков
// 1029 — не кратный размер (проверка вычисления сетки)
INSTANTIATE_TEST_SUITE_P(
    Sizes, VectorAddTest,
    ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029));
