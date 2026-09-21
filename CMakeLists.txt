cmake_minimum_required(VERSION 3.28)
project(hsys LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/deps/deps_ops.cmake)
include(cmake/deps/eigen.cmake)
include(cmake/deps/gtest.cmake)
include(cmake/deps/gbench.cmake)
include(cmake/utils/cuda_setup_targets.cmake)

add_subdirectory(work1)
