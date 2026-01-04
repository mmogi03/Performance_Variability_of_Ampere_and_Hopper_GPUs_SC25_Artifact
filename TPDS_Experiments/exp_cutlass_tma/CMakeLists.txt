cmake_minimum_required(VERSION 3.21)

project(wgmma_bench LANGUAGES CXX CUDA)

# Path to CUTLASS repo (pass -DCUTLASS_DIR=/path/to/cutlass)
set(CUTLASS_DIR "" CACHE PATH "Path to CUTLASS repo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Hopper SM90a by default (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90a)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(wgmma_sm90_bench       src/wgmma_sm90_bench.cu)
add_executable(wgmma_tma_sm90_bench   src/wgmma_tma_sm90_bench.cu)

set(COMMON_INCLUDES
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(wgmma_sm90_bench     PRIVATE ${COMMON_INCLUDES})
target_include_directories(wgmma_tma_sm90_bench PRIVATE ${COMMON_INCLUDES})

foreach(tgt IN ITEMS wgmma_sm90_bench wgmma_tma_sm90_bench)
  target_compile_options(${tgt} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda -lineinfo>
  )
  target_compile_definitions(${tgt} PRIVATE NDEBUG)
endforeach()

target_link_libraries(wgmma_sm90_bench     PRIVATE CUDA::cudart CUDA::curand CUDA::cuda_driver)
target_link_libraries(wgmma_tma_sm90_bench PRIVATE CUDA::cudart CUDA::curand CUDA::cuda_driver)
