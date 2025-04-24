message("   [Section] Building PYTHON BINDING ...")

cmake_minimum_required(VERSION 3.27)
# project(${PRJ_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(${PYBIND_NAME} unidock_engine.cpp)

target_link_libraries(${PYBIND_NAME} PRIVATE CUDA::cudart)
target_include_directories(
    ${PYBIND_NAME}
    PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/bin/unidock
)

# install(TARGETS unidock_engine DESTINATION .)

message("   [Section Done]  PYTHON BINDING !")