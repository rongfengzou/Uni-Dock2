//
// Created by Congcong Liu on 24-9-19.
//
#ifndef ERRORS_H
#define ERRORS_H

#include <string>
#include <cuda_runtime.h>
#include "spdlog/spdlog.h"

void check_cuda(cudaError_t cuda_err, char const *const func, const char *const file, int const line);
#define checkCUDA(val) check_cuda((val), #val, __FILE__, __LINE__)

void init_logger(const std::string& fp_log = "ud.log", int level=1);

#define CUDA_ERROR(...) printf("[CUDA error][%s:%d][%s][Block:%d,Thread:%d] ", \
__FILE__, __LINE__, __func__, \
blockIdx.x, threadIdx.x); \
printf(__VA_ARGS__); \
printf("\n")

#endif //ERRORS_H