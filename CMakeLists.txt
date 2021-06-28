cmake_minimum_required(VERSION 3.10)

project(vaccelrt-plugin-serrano
	VERSION 0.1
	DESCRIPTION "vAccelRT plugin for SERRANO project"
	LANGUAGES C CUDA)

# Export json with compile commands
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
IF (EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
	EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
		${CMAKE_SOURCE_DIR}/compile_commands.json
	)
ENDIF()

# CUDA installation paths
set(CUDA_DIR "/usr/local/cuda" CACHE STRING "Path to CUDA installation")
set(CUDA_INCLUDE ${CUDA_DIR}/include)
set(CUDA_LIB ${CUDA_DIR}/lib64)

link_directories(${CUDA_LIB})

set(VACCELRT_INCLUDE
	${CMAKE_SOURCE_DIR}/vaccelrt/src/include
	${CMAKE_SOURCE_DIR}/vaccelrt/third-party/slog/src
	CACHE STRING "Path to vAccel runtime headers")

set(SOURCES
    src/vaccel.c
    src/gemm.cu
)

# export rpath for libraries when installing
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(vaccel-serrano SHARED ${SOURCES})
target_include_directories(vaccel-serrano
	PRIVATE "src" ${VACCELRT_INCLUDE} ${CUDA_INCLUDE})

# Setup `make install`
include(GNUInstallDirs)
install(TARGETS vaccel-serrano DESTINATION "${CMAKE_INSTALL_LIBDIR}")
