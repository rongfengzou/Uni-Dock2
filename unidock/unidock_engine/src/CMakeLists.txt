find_package(CUDAToolkit REQUIRED)

set(sources
        format/json.cpp

        myutils/errors.cpp
        myutils/myio.cpp

        search/bvh.cpp

        score/gaff2.cpp

        task/DockTask.cpp

        cuda/bfgs_wrapper.cu
        cuda/cluster.cu
        cuda/mc.cu
        cuda/model.cu
        cuda/optimize.cu
        cuda/task.cu
        cuda/vina.cu
        cuda/common.cu

        screening.cpp
)

# compile a static lib for the bin to link to
add_library(${LIB_NAME} SHARED ${sources})

# include src directory
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  ${CMAKE_SOURCE_DIR}/include . )
target_link_libraries(${LIB_NAME} PUBLIC CUDA::cudart)