add_library(hsys_core INTERFACE)

target_include_directories(hsys_core INTERFACE include)

cuda_setup_targets(
  TARGETS hsys_core
  CUDA_MIN_ARCH 75
  FEATURES cxx_std_20
  LIBS Eigen3::Eigen
  SCOPE INTERFACE
)
