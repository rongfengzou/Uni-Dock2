//
// Created by Congcong Liu on 24-10-9.
//

#ifndef COMMON_CUH
#define COMMON_CUH

#include <curand_kernel.h>
#include "myutils/mymath.h"
#include "score/vina.h"
#include <cuda_runtime.h>



// -------------- Constants -----------------
extern __constant__ bool FLAG_CONSTRAINT_DOCK;
extern __constant__ Real BOX_X_HI;
extern __constant__ Real BOX_X_LO;
extern __constant__ Real BOX_Y_HI;
extern __constant__ Real BOX_Y_LO;
extern __constant__ Real BOX_Z_HI;
extern __constant__ Real BOX_Z_LO;
extern __constant__ Real TOR_PREC;
extern __constant__ Real BOX_PREC;
extern __constant__ Real PENALTY_SLOPE;

#if true
extern __constant__ Vina Score;
#else
extern __constant__ Gaff2 Score;
#endif

void init_constants(const DockParam& dock_param);


#define SCOPE_KERNEL __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
#define MAX_THREADS_PER_BLOCK 32
#define MIN_BLOCKS_PER_MP 32
const int TILE_SIZE = 32;
#define STRIDE_POSE 4
#define STRIDE_G 6



/**
 * @brief Generate an integer in [min, max] on CUDA
 * 
 * @param state CUDA random state
 * @param min Minimum value (int)
 * @param max Maximum value (int)
 * @return An integer in [min, max]
 */
__forceinline__ __device__ int gen_rand_int_within(curandStatePhilox4_32_10_t* state, int min, int max) {
    int range = max - min + 1;
    return min + (curand(state) % range);
}


/**
 * @brief Generate a real number in [min, max]
 * 
 * @param c A random integer within Integer range   
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @param n Number of divisions. The larger n is, the more accurate the result is.
 * @return A real number in [min, max]
 */
SCOPE_INLINE Real get_real_within_by_int(int c, Real min, Real max, int n=1001) {
    Real v = Real(c % n) * (max - min) / n + min;
    if (v < min){
        return min;
    }
    if(v > max){
        return max;
    }
    return v;
}



/**
 * @brief Generate a real number in [min, max] on CUDA
 * 
 * @param state CUDA random state
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @return A real number in [min, max]
 */
__forceinline__ __device__ Real gen_rand_real_within(curandStatePhilox4_32_10_t* state, Real min, Real max) {
    return get_real_within_by_int(curand(state), min, max);
}




__forceinline__ __device__ void gen_4_rand_in_sphere(Real *out_rand_4, curandStatePhilox4_32_10_t *state) {
    float4 rand_4;
    while (true) {  // on average, this will have to be run about twice
        rand_4 = curand_uniform4(state);  // ~ U[0,1]
        out_rand_4[0] = (rand_4.x - 0.5) * 2.0; // [-1, 1]
        out_rand_4[1] = (rand_4.y - 0.5) * 2.0;
        out_rand_4[2] = (rand_4.z - 0.5) * 2.0;

        if (norm_vec3(out_rand_4) < 1) { // not a specific length, also a randomization on rotation degree
            out_rand_4[3] = rand_4.w;
            return;
        }
    }
}

#endif //COMMON_CUH


