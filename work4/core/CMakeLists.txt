add_library(hsys_work4_core INTERFACE)

target_include_directories(hsys_work4_core INTERFACE include)

target_link_libraries(hsys_work4_core INTERFACE hsys_work3_core)
cuda_setup_targets(
  TARGETS hsys_work4_core
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
  LIBS Eigen3::Eigen
  SCOPE INTERFACE
)
