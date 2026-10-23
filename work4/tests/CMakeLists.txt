add_executable(hsys_tests_work4
  src/matrix_test.cu
)

set_target_properties(hsys_tests_work4 PROPERTIES
  OUTPUT_NAME "hsys_tests"
)

target_link_libraries(hsys_tests_work4 PRIVATE
  hsys_work4_core GTest::gtest_main Eigen3::Eigen
)

cuda_setup_targets(
  TARGETS hsys_tests_work4
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
)
