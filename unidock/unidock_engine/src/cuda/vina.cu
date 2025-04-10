//
// Created by Congcong Liu on 24-12-9.
//


#include "score/vina.h"

// ==================  Memory allocation for ONE object ==================
FlexParamVina* alloccp_FlexParamVina_gpu(const FlexParamVina& flex_param_vina, int natom){
    FlexParamVina* flex_param_vina_cu;
    cudaMalloc(&flex_param_vina_cu, sizeof(FlexParamVina));

    FlexParamVina flex_param_vina_tmp;
    flex_param_vina_tmp.npair_intra = flex_param_vina.npair_intra;
    flex_param_vina_tmp.npair_inter = flex_param_vina.npair_inter;
    cudaMalloc(&flex_param_vina_tmp.pairs_intra, sizeof(int) * flex_param_vina.npair_intra * 2);
    cudaMemcpy(flex_param_vina_tmp.pairs_intra, flex_param_vina.pairs_intra, sizeof(int) * flex_param_vina.npair_intra * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&flex_param_vina_tmp.r1_plus_r2_intra, sizeof(Real) * flex_param_vina.npair_intra);
    cudaMemcpy(flex_param_vina_tmp.r1_plus_r2_intra, flex_param_vina.r1_plus_r2_intra, sizeof(Real) * flex_param_vina.npair_intra, cudaMemcpyHostToDevice);
    cudaMalloc(&flex_param_vina_tmp.pairs_inter, sizeof(int) * flex_param_vina.npair_inter * 2);
    cudaMemcpy(flex_param_vina_tmp.pairs_inter, flex_param_vina.pairs_inter, sizeof(int) * flex_param_vina.npair_inter * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&flex_param_vina_tmp.r1_plus_r2_inter, sizeof(Real) * flex_param_vina.npair_inter);
    cudaMemcpy(flex_param_vina_tmp.r1_plus_r2_inter, flex_param_vina.r1_plus_r2_inter, sizeof(Real) * flex_param_vina.npair_inter, cudaMemcpyHostToDevice);
    cudaMalloc(&flex_param_vina_tmp.atom_types, sizeof(int) * natom);
    cudaMemcpy(flex_param_vina_tmp.atom_types, flex_param_vina.atom_types, sizeof(int) * natom, cudaMemcpyHostToDevice);

    cudaMemcpy(flex_param_vina_cu, &flex_param_vina_tmp, sizeof(FlexParamVina), cudaMemcpyHostToDevice);
    return flex_param_vina_cu;
}
void free_FlexParamVina_gpu(FlexParamVina* flex_param_vina_cu){
    FlexParamVina flex_param_vina_tmp;
    cudaMemcpy(&flex_param_vina_tmp, flex_param_vina_cu, sizeof(FlexParamVina), cudaMemcpyDeviceToHost);
    cudaFree(flex_param_vina_tmp.atom_types);
    cudaFree(flex_param_vina_tmp.pairs_intra);
    cudaFree(flex_param_vina_tmp.r1_plus_r2_intra);
    cudaFree(flex_param_vina_tmp.pairs_inter);
    cudaFree(flex_param_vina_tmp.r1_plus_r2_inter);
    cudaFree(flex_param_vina_cu);
}


FixParamVina* alloccp_FixParamVina_gpu(const FixParamVina& fix_param_vina, int natom){
    FixParamVina* fix_param_vina_cu;
    cudaMalloc(&fix_param_vina_cu, sizeof(FixParamVina));

    FixParamVina fix_param_vina_tmp;
    cudaMalloc(&fix_param_vina_tmp.atom_types, sizeof(int) * natom);
    cudaMemcpy(fix_param_vina_tmp.atom_types, fix_param_vina.atom_types, sizeof(int) * natom, cudaMemcpyHostToDevice);

    cudaMemcpy(fix_param_vina_cu, &fix_param_vina_tmp, sizeof(FixParamVina), cudaMemcpyHostToDevice);
    return fix_param_vina_cu;
}
void free_FixParamVina_gpu(FixParamVina* fix_param_vina_cu){
    FixParamVina fix_param_vina_tmp;
    cudaMemcpy(&fix_param_vina_tmp, fix_param_vina_cu, sizeof(FixParamVina), cudaMemcpyDeviceToHost);
    cudaFree(fix_param_vina_tmp.atom_types);
    cudaFree(fix_param_vina_cu);
}
