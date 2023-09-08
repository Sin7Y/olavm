cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_lib LANGUAGES CXX CUDA)

if (DEFINED CMAKE_BUILD_TYPE)
    string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
else()
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
if(NOT CUDAToolkit_FOUND)
    include(FindCUDAToolkit)
endif()
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

check_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    set(NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets)
    message(STATUS "${PROJECT_NAME} compile cuda code in release mode")
else()
    set(NVCC_FLAGS -G -Wno-deprecated-gpu-targets)
    message(STATUS "${PROJECT_NAME} compile cuda code in debug mode")
endif()

file(GLOB_RECURSE source_file LIST_DIRECTORIES false src/*.cpp src/*.c src/*.cuh src/*.h src/*.cu)
add_library(${PROJECT_NAME} STATIC ${source_file})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        ${NVCC_FLAGS}
        >)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

install(TARGETS ${PROJECT_NAME})