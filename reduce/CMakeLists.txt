cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(reduce LANGUAGES CXX CUDA)

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
add_executable(reduce ${CPP_SOURCES} ${CU_SOURCES} ${HEADER_FILES})


# switch Debug/Release
if (TYPE STREQUAL "debug")
    message("Debug mode:")
    add_definitions(-DDEBUG)
    target_compile_options(reduce PRIVATE -g -O0 -lineinfo)
elseif(TYPE STREQUAL "release")
    message("Release mode:")
    target_compile_options(reduce PRIVATE -O3)
endif()