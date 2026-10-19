add_executable(hsys_benchmarks_work2
  src/matrix_benchmark.cu
)

target_link_libraries(hsys_benchmarks_work2 PRIVATE hsys_work2_core benchmark::benchmark_main Eigen3::Eigen)

cuda_setup_targets(
  TARGETS hsys_benchmarks_work2
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
)
