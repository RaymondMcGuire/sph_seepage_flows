# CMake Version
cmake_minimum_required(VERSION 3.26.0)

# Project Name
project(sph_seeage_flow LANGUAGES C CXX CUDA)

# For Visual Studio
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	# Optimization options for C/C++
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /wd4819 /wd4068 /wd26812 /wd26495 /O2 /Ob2 /Ot /Oi /Gy /MP")
endif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")


set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Detect host GPUs" FORCE)

find_package(CUDAToolkit REQUIRED)

# Sub Projects
add_subdirectory(math)
add_subdirectory(seepage_flow_cuda)
add_subdirectory(seepage_flow)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT seepage_flow)