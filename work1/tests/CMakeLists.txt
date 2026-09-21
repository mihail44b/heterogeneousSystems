add_executable(hsys_tests
  src/vector_test.cu
)

target_link_libraries(hsys_tests PRIVATE hsys_core GTest::gtest_main)

cuda_setup_targets(
  TARGETS hsys_tests
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
)

gtest_discover_tests(hsys_tests)
