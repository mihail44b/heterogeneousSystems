add_executable(hsys_benchmarks_work3
  src/matrix_benchmark.cu
)

set_target_properties(hsys_benchmarks_work3 PROPERTIES
OUTPUT_NAME "hsys_benchmarks")

target_link_libraries(hsys_benchmarks_work3 PRIVATE
hsys_work3_core benchmark::benchmark_main Eigen3::Eigen)

cuda_setup_targets(
  TARGETS hsys_benchmarks_work3
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
)
