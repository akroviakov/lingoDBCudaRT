cmake_minimum_required(VERSION 3.20)
project(cuda_exp LANGUAGES CXX CUDA)

option(USE_GALLATIN "Use gallatin library for memory allocation" ON)

if(USE_GALLATIN)
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME gallatin
    GITHUB_REPOSITORY saltsystemslab/gallatin
    GIT_TAG origin/main
  )
  add_definitions(-DGALLATIN_ENABLED) 
  message(STATUS "Gallatin enabled.")
else()
  message(STATUS "Gallatin disabled.")
endif()


if(DEFINED INITIAL_CAPACITY)
    get_property(INITIAL_CAPACITY_VALUE CACHE INITIAL_CAPACITY PROPERTY VALUE)
    message(STATUS "INITIAL_CAPACITY is set to ${INITIAL_CAPACITY_VALUE}")
else()
    set(INITIAL_CAPACITY 1024)
    message(STATUS "INITIAL_CAPACITY is not set, defaulting to ${INITIAL_CAPACITY}")
endif()
set(CMAKE_CUDA_ARCHITECTURES "native")
set(COMMON_COMPILE_FEATURES cuda_std_17)
set(COMMON_COMPILE_OPTIONS_CXX -O3 -g -Wall)
set(COMMON_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(COMMON_DEFINITIONS PRIVATE INITIAL_CAPACITY=${INITIAL_CAPACITY})

# add_subdirectory(lib)
add_subdirectory(lingodbDSA)
