cmake_minimum_required(VERSION 3.20.0)
project(p4mlir LANGUAGES CXX)

add_subdirectory(third_party)

set(P4MLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(P4MLIR_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${P4MLIR_SOURCE_DIR}/include)
include_directories(${P4MLIR_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
