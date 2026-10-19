#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <random>
#include <tuple>
#include <vector>

#include <work2/entities/matrix.hpp>
#include <work3/functions/matrix_multiply.hpp>

namespace hsys::work3::tests {

using work2::core::Matrix;
using work3::core::operator*;

class MatrixMultiplyParamTest
    : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {
};

TEST_P(MatrixMultiplyParamTest, MatchesEigen) {
  const auto [m, n, k] = GetParam();

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> a_host(m * k);
  for (auto& val : a_host) {
    val = dist(gen);
  }

  std::vector<float> b_host(k * n);
  for (auto& val : b_host) {
    val = dist(gen);
  }

  Matrix<float> a_gpu(m, k, a_host);
  Matrix<float> b_gpu(k, n, b_host);

  Matrix<float> c_gpu = a_gpu * b_gpu;
  std::vector<float> c_host = c_gpu.to_host();

  using RowMatrixXf
      = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Eigen::Map<const RowMatrixXf> a_eigen(
      a_host.data(), static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
  Eigen::Map<const RowMatrixXf> b_eigen(
      b_host.data(), static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));

  RowMatrixXf c_eigen = a_eigen * b_eigen;

  Eigen::Map<const RowMatrixXf> c_gpu_map(
      c_host.data(), static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));

  EXPECT_TRUE(c_gpu_map.isApprox(c_eigen, 1e-5f))
      << "Mismatch for matrix sizes: m=" << m << ", n=" << n << ", k=" << k;
}

TEST(MatrixMultiplyExceptionTest, IncompatibleDimensionsThrows) {
  Matrix<float> a(2, 3);
  Matrix<float> b(4, 5);

  EXPECT_THROW({ [[maybe_unused]] auto c = a * b; }, std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(SweepDimensions,
    MatrixMultiplyParamTest,
    ::testing::Combine(::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512),
        ::testing::Values(1, 2, 3, 127, 128, 129, 512)));

}  // namespace hsys::work3::tests
