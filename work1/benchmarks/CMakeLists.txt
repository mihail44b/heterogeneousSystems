add_executable(hsys_benchmarks
  src/vector_benchmark.cu
)

target_link_libraries(hsys_benchmarks PRIVATE hsys_core benchmark::benchmark_main Eigen3::Eigen)

cuda_setup_targets(
  TARGETS hsys_benchmarks
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
)
