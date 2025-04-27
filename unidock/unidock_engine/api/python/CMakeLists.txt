message("      [Sub-Section] Building PYTHON BINDING ...")

set(PYBIND_NAME pipeline)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(${PYBIND_NAME} pipeline.cpp)

target_link_libraries(${PYBIND_NAME} PRIVATE ${LIB_NAME})
target_include_directories(${PYBIND_NAME}
    PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

message("      [Sub-Section Done]  PYTHON BINDING !")