add_executable(hsys_tests_work3
  src/matrix_test.cu
)

# Задаем стандартное выходное имя бинарника: build/work3/tests/hsys_tests
set_target_properties(hsys_tests_work3 PROPERTIES OUTPUT_NAME "hsys_tests")

target_link_libraries(hsys_tests_work3 PRIVATE
hsys_work3_core GTest::gtest_main)

cuda_setup_targets(
  TARGETS hsys_tests_work3
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
)

gtest_discover_tests(hsys_tests_work3)
