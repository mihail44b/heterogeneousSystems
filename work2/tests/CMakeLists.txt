add_executable(hsys_tests_work2
  src/matrix_test.cu
)

target_link_libraries(hsys_tests_work2 PRIVATE hsys_work2_core GTest::gtest_main)

cuda_setup_targets(
  TARGETS hsys_tests_work2
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
)

gtest_discover_tests(hsys_tests_work2)
