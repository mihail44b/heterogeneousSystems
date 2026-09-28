core_inc = include_directories(
  'include',
  'src',
)

core_sources = files(
)

core_lib = static_library(
  'work1_core',
  core_sources,
  include_directories: core_inc,
  cpp_args: [
    '-std=c++20',
  ],
  install: false,
)

core_dep = declare_dependency(
  link_with: core_lib,
  include_directories: core_inc,
)
