### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -x cu -g -lcurand
    -std=c++11
    )


# Specify include directories
include_directories(
    headers
    kernels
    )

# Specify library paths
link_directories(
 /usr/local/cuda-9.0/targets/x86_64-linux/lib/
    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    gillespie
    gillespie.cu
  
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
      gillespie
      curand

    )
