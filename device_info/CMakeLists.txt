cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(device_info LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Include directories for header files
include_directories(
    ${CMAKE_SOURCE_DIR}/common
    ${CUDA_INCLUDE_DIRS}
)

# Collect all source files
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp" "common/*.cpp")
file(GLOB_RECURSE CU_SOURCES "src/*.cu" "common/*.cu")
file(GLOB_RECURSE HEADER_FILES "common/*.hpp" "common/*.h" "common/*.cuh")

# Create executable
add_executable(device_info ${CPP_SOURCES} ${CU_SOURCES} ${HEADER_FILES})


# switch Debug/Release
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode:")
    add_definitions(-DDEBUG)
    target_compile_options(device_info PRIVATE -g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Release mode:")
    target_compile_options(device_info PRIVATE -O3)
endif()