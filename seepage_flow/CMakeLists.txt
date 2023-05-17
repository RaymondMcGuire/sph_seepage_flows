# Project Name
set(PROJECT_NAME seepage_flow)

# Find Extra Library Package
set(EXTLIBS_INCLUDE ${CMAKE_SOURCE_DIR}/extlib/include)
set(EXTLIBS_SRC ${CMAKE_SOURCE_DIR}/extlib/src)
set(EXTLIBS_LIB ${CMAKE_SOURCE_DIR}/extlib/lib)
link_directories(${EXTLIBS_LIB})

set(VTK_DIR ${CMAKE_SOURCE_DIR}/extlib/vtk/lib/cmake/vtk-9.2)
find_package(VTK REQUIRED)
set(DLL_STRING "${CMAKE_SOURCE_DIR}/extlib/vtk/bin")


# Kiri PBS CUDA Library
set(KIRI_PBS_CUDA_LIB_INLCUDE ${CMAKE_BINARY_DIR}/seepage_flow_cuda/include)

# Kiri Math Library
set(KIRI_MATH_LIB_INLCUDE ${CMAKE_SOURCE_DIR}/math/include)

# Configuration File
configure_file(${CMAKE_SOURCE_DIR}/configuration/root_directory.h.in  ${CMAKE_BINARY_DIR}/configuration/root_directory.h)
set(CONFIGURATION_INCLUDE ${CMAKE_BINARY_DIR}/configuration)

# Extra Static Library
set(EXTLIBS_LINK_LIBS_DEBUG seepage_flow_cuda)

# Project Config
file(GLOB_RECURSE SOURCES  "src/*.cpp" "src/*.c")
file(GLOB_RECURSE HEADERS  "include/*.hpp" "include/*.h")
set(PROJECT_FILES ${SOURCES} ${HEADERS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ../bin/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ../bin/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ../lib/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ../lib/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ../lib/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ../lib/Debug)

add_executable(${PROJECT_NAME} ${PROJECT_FILES})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${PROJECT_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
    include
    ${EXTLIBS_INCLUDE}
    ${VTK_USE_FILE}
    ${CONFIGURATION_INCLUDE}
    ${KIRI_PBS_CUDA_LIB_INLCUDE}
    ${KIRI_MATH_LIB_INLCUDE}
)

target_link_libraries(${PROJECT_NAME} ${EXTLIBS_LINK_LIBS_DEBUG} ${VTK_LIBRARIES})

set(WD_DEBUG_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Debug/)
set(WD_RELEASE_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Release/)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME_DEBUG ${PROJECT_NAME}d
    OUTPUT_NAME_RELEASE ${PROJECT_NAME}
    VS_DEBUGGER_WORKING_DIRECTORY "$<$<CONFIG:debug>:${WD_DEBUG_FILE_PATH}>$<$<CONFIG:release>:${WD_RELEASE_FILE_PATH}>"
)
# Copy DLLs
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${DLL_STRING}" $<TARGET_FILE_DIR:${PROJECT_NAME}>)