# Project Name
set(PROJECT_NAME kiri_examples)

# Find Extra Library Package
set(EXTLIBS_INCLUDE ${CMAKE_SOURCE_DIR}/KiriExtlib/include)
set(EXTLIBS_SRC ${CMAKE_SOURCE_DIR}/KiriExtlib/src)
set(EXTLIBS_LIB ${CMAKE_SOURCE_DIR}/KiriExtlib/lib)
link_directories(${EXTLIBS_LIB})

if(WIN32)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/)
    message(STATUS "CMAKE_MODULE_PATH   : ${CMAKE_MODULE_PATH}")
    find_package(GLFW3 REQUIRED)
    message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")
    find_package(ASSIMP REQUIRED)
    message(STATUS "Found ASSIMP in ${ASSIMP_INCLUDE_DIR}")
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        message(STATUS "OpenMP_C_Found   : ${OpenMP_C_FOUND}")
        message(STATUS "OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Stb_image Library
add_library(stb_image ${EXTLIBS_SRC}/stb_image.cpp ${EXTLIBS_SRC}/stb_image_write.cpp)
target_include_directories(stb_image PUBLIC ${EXTLIBS_INCLUDE})

# Glad Library
add_library(glad ${EXTLIBS_SRC}/glad.c)
target_include_directories(glad PUBLIC ${EXTLIBS_INCLUDE})

# Imgui Library
set (IMGUI_SOURCE_DIR ${EXTLIBS_INCLUDE}/imgui/source)
set (IMGUI_INCLUDE_DIR ${EXTLIBS_INCLUDE}/imgui/include)
file(GLOB_RECURSE IMGUI_FILES
	"${IMGUI_INCLUDE_DIR}/*.h"
    "${IMGUI_SOURCE_DIR}/*.cpp"
)
add_library(imgui STATIC ${IMGUI_FILES})
target_include_directories(imgui PUBLIC ${IMGUI_INCLUDE_DIR} ${EXTLIBS_INCLUDE})

# Kiri PBS CUDA Library
set(KIRI_PBS_CUDA_LIB_INLCUDE ${CMAKE_BINARY_DIR}/KiriPBSCuda/include)

# Kiri Math Library
set(KIRI_MATH_LIB_INLCUDE ${CMAKE_SOURCE_DIR}/KiriMath/include)

# Configuration File
configure_file(${CMAKE_SOURCE_DIR}/configuration/root_directory.h.in  ${CMAKE_BINARY_DIR}/configuration/root_directory.h)
set(CONFIGURATION_INCLUDE ${CMAKE_BINARY_DIR}/configuration)

# Kiri Core Library
set(KIRI_CORE_LIB_INLCUDE ${CMAKE_SOURCE_DIR}/KiriCore/include)

# Extra Static Library
set(EXTLIBS_LINK_LIBS_DEBUG glfw3 opengl32 assimp-vc141-mt glad stb_image imgui kiri_pbs_cuda kiri_math kiri_core)

# Macro Command
macro(makeLink src dest target)
  add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest}  DEPENDS  ${dest} COMMENT "mklink ${src} -> ${dest}")
endmacro()

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
    ${CONFIGURATION_INCLUDE}
    ${KIRI_PBS_CUDA_LIB_INLCUDE}
    ${KIRI_MATH_LIB_INLCUDE}
    ${KIRI_CORE_LIB_INLCUDE}
)

target_link_libraries(${PROJECT_NAME} ${EXTLIBS_LINK_LIBS_DEBUG} debug partiod optimized partio)

set(WD_DEBUG_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Debug/)
set(WD_RELEASE_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Release/)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME_DEBUG ${PROJECT_NAME}d
    OUTPUT_NAME_RELEASE ${PROJECT_NAME}
    VS_DEBUGGER_WORKING_DIRECTORY "$<$<CONFIG:debug>:${WD_DEBUG_FILE_PATH}>$<$<CONFIG:release>:${WD_RELEASE_FILE_PATH}>"
)

# Copy Shaders
file(GLOB_RECURSE SHADERS
 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vs
 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.fs
 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.gs
)

foreach(shader ${SHADERS})
    if(WIN32)
        get_filename_component(shader_name "${shader}" NAME) 
        set(SHADER_DEBUG_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Debug/shader/${shader_name})
        set(SHADER_RELEASE_FILE_PATH ${CMAKE_BINARY_DIR}/bin/Release/shader/${shader_name})
        add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${shader}"  "$<$<CONFIG:debug>:${SHADER_DEBUG_FILE_PATH}>$<$<CONFIG:release>:${SHADER_RELEASE_FILE_PATH}>")
    elseif(UNIX AND NOT APPLE)
        file(COPY ${shader} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin/${PROJECT_NAME})
    elseif(APPLE)
        # create symbolic link for *.vs *.fs *.gs
        get_filename_component(SHADERNAME ${shader} SHADER_FOLDER_NAME)
        makeLink(${shader} ${CMAKE_CURRENT_BINARY_DIR}/output/${PROJECT_NAME}/${SHADERNAME} ${PROJECT_NAME})
    endif(WIN32)
endforeach(shader)

#Copy DLLs
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/KiriExtlib/dlls" $<TARGET_FILE_DIR:${PROJECT_NAME}>)
