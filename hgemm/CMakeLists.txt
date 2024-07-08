cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(hgemm LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

# Include directories for header files
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CUDA_INCLUDE_DIRS}
)

# Collect all source files
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "common/*.cpp")
file(GLOB_RECURSE CU_SOURCES "src/*.cu" "common/*.cu")
file(GLOB_RECURSE HEADER_FILES "include/*.h" "include/*.hpp" "include/*.cuh" "common/*.hpp" "common/*.h" "common/*.cuh")

# Create executable
add_executable(hgemm ${CPP_SOURCES} ${CU_SOURCES} ${HEADER_FILES})

# 链接cuBLAS库
target_link_libraries(hgemm ${CUDA_cublas_LIBRARY} )

# switch Debug/Release
if (TYPE STREQUAL "Debug")
    message("Debug mode:")
    add_definitions(-DDEBUG)
    target_compile_options(hgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -O0 -lineinfo --generate-code arch=compute_86,code=sm_86>)
elseif(TYPE STREQUAL "Release")
    message("Release mode:")
    target_compile_options(hgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-code arch=compute_86,code=sm_86>)
endif()