# CMake Version
cmake_minimum_required(VERSION 3.18.0)

# Project Name
project(sph_seeage_flows LANGUAGES C CXX CUDA)

# Get CUDA INFO
find_package(CUDA)
if (CUDA_FOUND)
  set(CUDA_ARCH_DYNAMIC "FALSE")
  set(CUDA_FIXED_ARCH "75")
  #Get CUDA compute capability
  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkCUDA/compile/cuda_script)
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/checkCUDA/check_cuda.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE} -Xcompiler "/wd4819")
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    if (${CUDA_ARCH_DYNAMIC})
      message(STATUS "CUDA Architecture: ${ARCH}")
    else()
      message(STATUS "CUDA Architecture: ${CUDA_FIXED_ARCH}")
    endif()

    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  else()
    message(WARNING ${ARCH})
  endif()
endif()

# For Visual Studio
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	# Optimization options for C/C++
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /wd4819 /wd26812 /wd26495 /O2 /Ob2 /Ot /Oi /Gy /MP")
endif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")


# Compile Type
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

if (${CUDA_ARCH_DYNAMIC})
  set(CMAKE_CUDA_ARCHITECTURES ${ARCH})
else()
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_FIXED_ARCH})
endif()


# Sub Projects
add_subdirectory(KiriMathMini)
add_subdirectory(seepage_flows_cuda)
add_subdirectory(seepage_flows)