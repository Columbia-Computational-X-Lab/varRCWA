cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
# ===============================================
# Variant RCWA
# ===============================================
# Developed by 
# Ziwei Zhu @ Columbia University
#
# How to generate Makefile:
# cmake -D CMAKE_CUDA_HOST_COMPILER=gcc -D CMAKE_CUDA_COMPILER=nvcc -D CMAKE_CXX_COMPILER=g++ .
# ===============================================
option(WITH_OPENMP "enable OpenMP acceleration or not" ON)
option(BUILD_DEBUG "Turn on the debug mode" OFF)
option(BUILD_TESTS "Build unit test cases" ON)
option(BUILD_GPU "Build GPU examples" ON)

if (BUILD_GPU)
  Project(varRCWA LANGUAGES CUDA CXX C)
else ()
  Project(varRCWA LANGUAGES CXX C)
endif()

# General Configuration: find all the required libraries.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(sploosh_utils)
# compile option

set(CMAKE_CXX_FLAGS "-march=native -Wall -Wno-psabi -Wno-sign-compare -std=c++2a -Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if ( BUILD_DEBUG )
    add_definitions(-DDEBUG)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
else ()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()

#===================================================================
set(CMAKE_BUILD_TYPE Release)

set(ROOT_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "Root build directory")
set(ROOT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE STRING "Root source directory")
set(ROOT_BIN_DIR "${ROOT_BUILD_DIR}/bin" CACHE STRING "Root binary directory")
set(ROOT_TESTS_DIR "${ROOT_BUILD_DIR}/tests" CACHE STRING "Root tests directory")

file(GLOB RCWA_SRC_FILES 
    "${ROOT_SRC_DIR}/rcwa/*.h"
    "${ROOT_SRC_DIR}/rcwa/*.cpp")

file(GLOB CORE_SRC_FILES
    "${ROOT_SRC_DIR}/core/*.h"
    "${ROOT_SRC_DIR}/core/*.cpp")

file(GLOB GDSTK_SRC_FILES
    "${ROOT_SRC_DIR}/gdstk/*.h"
    "${ROOT_SRC_DIR}/gdstk/*.cpp"
    "${ROOT_SRC_DIR}/gdstk/clipperlib/*.hpp"
    "${ROOT_SRC_DIR}/gdstk/clipperlib/*.cpp")

file(GLOB GPU_SRC_FILES
  "${ROOT_SRC_DIR}/gpu/*.cuh"
  "${ROOT_SRC_DIR}/gpu/*.h"
  "${ROOT_SRC_DIR}/gpu/*.cu"
  "${ROOT_SRC_DIR}/gpu/*.cpp")

mark_as_advanced(ROOT_BUILD_DIR)
mark_as_advanced(ROOT_SRC_DIR)
mark_as_advanced(ROOT_BIN_DIR)
mark_as_advanced(ROOT_TESTS_DIR)

#===================================================================
if (WITH_OPENMP)
	find_package(OpenMP)
	if (OPENMP_FOUND)
	    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	endif ()
endif ()

find_package(MKL REQUIRED)
if(MKL_FOUND)
    message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
    message(STATUS "MKL include directory: ${MKL_INCLUDE_DIR}")
    message(STATUS "MKL OpenMP type: ${MKL_OPENMP_TYPE}")
    message(STATUS "MKL OpenMP library: ${MKL_OPENMP_LIBRARY}")
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif()

find_package(Eigen3 3.3 CONFIG REQUIRED)
if (EIGEN3_FOUND)
   message(STATUS "Found Eigen version ${EIGEN3_VERSION_STRING}: ${EIGEN3_INCLUDE_DIRS}")
endif ()

find_package(TBB 2021.5 REQUIRED)
if (TBB_FOUND)
    message(STATUS "Found Intel TBB version ${TBB_VERSION}: ${TBB_INCLUDE_DIRS} ${TBB_LIBRARIES}")
endif ()

if (BUILD_GPU)
  set(MAGMA_INCLUDE_DIR /usr/local/include)
  set(MAGMA_LIBRARIES libmagma.so;
      libiomp5.so)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/src)

if (BUILD_TESTS)
    add_subdirectory(src/tests)
endif()