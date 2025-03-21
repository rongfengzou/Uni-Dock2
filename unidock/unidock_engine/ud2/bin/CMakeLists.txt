message("   [Section] Building BIN ...")

add_executable(${BIN_NAME}
        unidock/screening.h
        unidock/screening.cpp
        unidock/main.cpp
)
target_compile_definitions(${BIN_NAME} PRIVATE VERSION_NUMBER="${VERSION_NUMBER}")
target_link_libraries(${BIN_NAME} PRIVATE yaml-cpp_0.8.0 ${LIB_NAME})
target_include_directories(${BIN_NAME}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)

# ---------------------  For Test  ---------------------
add_executable(myapp
        myapp/myapp.cu
)
target_link_libraries(myapp PRIVATE cudart_static)
target_include_directories(myapp PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ---------------------  For Test  ---------------------
add_executable(myapp_cxx
        myapp/myapp.cpp
)

message("   [Section Done]  BIN !")
