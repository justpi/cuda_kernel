cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(transpose LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

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
add_executable(transpose ${CPP_SOURCES} ${CU_SOURCES} ${HEADER_FILES})

# 链接cuBLAS库
target_link_libraries(transpose ${CUDA_cublas_LIBRARY})

# switch Debug/Release
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode:")
    add_definitions(-DDEBUG)
    target_compile_options(transpose PRIVATE -g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Release mode:")
    target_compile_options(transpose PRIVATE -O3)
endif()