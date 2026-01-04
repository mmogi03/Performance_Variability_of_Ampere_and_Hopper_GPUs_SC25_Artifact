cmake_minimum_required(VERSION 3.20)

project(BF16 LANGUAGES CUDA CXX)

# -------------------------------
# CUDA architecture
# -------------------------------
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA SM architecture")

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------
# Executable
# -------------------------------
add_executable(bf16
  BF16/src/main.cu
)

# -------------------------------
# Include paths (from your Makefile)
# -------------------------------
target_include_directories(bf16 PRIVATE
  /home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include
)

# -------------------------------
# Library paths + libs (Makefile style)
# -------------------------------
target_link_directories(bf16 PRIVATE
  /home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64
)

target_link_libraries(bf16
  cublas
  curand
)

# -------------------------------
# NVCC flags (optional but safe)
# -------------------------------
target_compile_options(bf16 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(bf16 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(bf16 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/BF16/src
  OUTPUT_NAME main
)
