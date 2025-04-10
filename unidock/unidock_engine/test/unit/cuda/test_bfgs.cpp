//
// Created by Congcong Liu on 24-12-9.
//

#include <algorithm>
#include <catch2/catch_amalgamated.hpp>
#include "optimize/bfgs.h"


void prepare_param_by_topo(const FlexTopo* flex_topo, const FixMol* fix_mol, const FixParamVina* fix_param, FlexParamVina* flex_param){
    flex_param->npair_intra = flex_topo->natom * (flex_topo->natom - 1) / 2;
    if (flex_param->npair_intra > 0){
        flex_param->pairs_intra = new int[flex_param->npair_intra * 2];
        flex_param->r1_plus_r2_intra = new Real[flex_param->npair_intra];
    }else{
        flex_param->pairs_intra = nullptr;
        flex_param->r1_plus_r2_intra = nullptr;
    }
    flex_param->atom_types = new int[flex_topo->natom];
    int ipair = 0;
    for (int i = 0; i < flex_topo->natom; i++){
        for (int j = i + 1; j < flex_topo->natom; j++){
            flex_param->pairs_intra[ipair * 2] = i;
            flex_param->pairs_intra[ipair * 2 + 1] = j;
            flex_param->r1_plus_r2_intra[ipair] = VN_VDW_RADII[flex_topo->vn_types[i]] + VN_VDW_RADII[flex_topo->vn_types[j]];
            ipair++;
        }
        flex_param->atom_types[i] = flex_topo->vn_types[i];
    }

    flex_param->npair_inter = flex_topo->natom * fix_mol->natom;
    flex_param->pairs_inter = new int[flex_param->npair_inter * 2];
    flex_param->r1_plus_r2_inter = new Real[flex_param->npair_inter];
    ipair = 0;
    for (int i = 0; i < flex_topo->natom; i++){
        for (int j = 0; j < fix_mol->natom; j++){
            flex_param->pairs_inter[ipair * 2] = i;
            flex_param->pairs_inter[ipair * 2 + 1] = j;
            flex_param->r1_plus_r2_inter[ipair] = VN_VDW_RADII[flex_param->atom_types[i]] + VN_VDW_RADII[fix_param->atom_types[j]];
            ipair++;
        }
    }
}

void prepare_flex_1(FixMol* fix_mol, FixParamVina* fix_param, FlexParamVina* flex_param,
    FlexPose* x, FlexTopo* flex_topo, FlexPose* out_x_new, FlexPoseGradient* out_g_new,
    Real r = 1.1){

    fix_mol->natom = 1;
    fix_mol->coords = new Real[3];
    fix_mol->coords[0] = 0.;
    fix_mol->coords[1] = 0.;
    fix_mol->coords[2] = 0.;
    fix_param->atom_types = new int[1];
    fix_param->atom_types[0] = VN_TYPE_C_H;

    flex_topo->natom = 1;
    flex_topo->vn_types = new int[1];
    flex_topo->vn_types[0] = VN_TYPE_C_H;

    prepare_param_by_topo(flex_topo, fix_mol, fix_param, flex_param);

    x->energy = 999;
    x->center[0] = r;
    x->center[1] = 0.;
    x->center[2] = 0.;
    x->coords = new Real[3];
    x->coords[0] = r;
    x->coords[1] = 0.;
    x->coords[2] = 0.;
    x->dihedrals = nullptr;

    out_x_new->energy = 999;
    out_x_new->coords = new Real[3];
    std::fill(out_x_new->coords, out_x_new->coords + 3, 0.);
    out_x_new->dihedrals = nullptr;

    out_g_new->dihedrals_g = nullptr;
}




TEST_CASE("test bfgs process", "[bfgs]"){
    Real out_e = 9999, out_alpha = -9999;
    FixMol fix_mol;
    FixParamVina fix_param;
    FlexParamVina flex_param;
    FlexPose x;
    FlexTopo flex_topo;
    FlexPose out_x_new;
    FlexPoseGradient out_g_new;
    Vina vina;
    Real f_ref = 0;
    Real e_ref = 0;

    //--------------- Test 1 -----------------
    // One Carbon as receptor, one Carbon as flex
    for (Real r = 0; r < 2; r += 0.5){
        // Set Receptor
        prepare_flex_1(&fix_mol, &fix_param, &flex_param, &x, &flex_topo, &out_x_new, &out_g_new, r);

        cal_e_grad_one_gpu(fix_mol, fix_param, flex_param, x, flex_topo, &out_g_new, &out_e);
        e_ref = vina.eval_ef(r - flex_param.r1_plus_r2_inter[0], fix_param.atom_types[0], flex_param.atom_types[0], &f_ref);
        REQUIRE_THAT(out_e, Catch::Matchers::WithinAbs(e_ref, 1e-4));
        if (r == 0.){
            f_ref = 0.; // too close, the calculation generates wrong result
        }
        REQUIRE_THAT(out_g_new.center_g[0], Catch::Matchers::WithinAbs(-f_ref, 1e-4));
        REQUIRE_THAT(out_g_new.center_g[1], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.center_g[2], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.orientation_g[0], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.orientation_g[1], Catch::Matchers::WithinAbs(1, 1e-4));
        REQUIRE_THAT(out_g_new.orientation_g[2], Catch::Matchers::WithinAbs(0, 1e-4));
        REQUIRE_THAT(out_g_new.orientation_g[3], Catch::Matchers::WithinAbs(0, 1e-4));
    }

    // r == 1.5, record the g
    Real g[3] = { -out_g_new.center_g[0], -out_g_new.center_g[1], -out_g_new.center_g[2] };
    line_search_one_gpu(fix_mol, fix_param,
        flex_param, x,
        flex_topo,
        &out_x_new, &out_g_new, &out_e, &out_alpha);

    //fixme: put bug data as input: test/dev1/;

    REQUIRE_THAT(out_alpha, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(out_x_new.coords[0], Catch::Matchers::WithinAbs(x.coords[0] + out_alpha * g[0], 1e-4));
    REQUIRE_THAT(out_x_new.coords[1], Catch::Matchers::WithinAbs(x.coords[1] + out_alpha * g[1], 1e-4));
    REQUIRE_THAT(out_x_new.coords[2], Catch::Matchers::WithinAbs(x.coords[2] + out_alpha * g[2], 1e-4));
    Real r = norm_vec3(out_x_new.coords);
    REQUIRE_THAT(out_e, Catch::Matchers::WithinAbs(vina.eval_ef(r - flex_param.r1_plus_r2_inter[0], fix_param.atom_types[0], flex_param.atom_types[0], &f_ref), 1e-4));
    //![bfgs]
}


void prepare_flex_2(FixMol* fix_mol, FixParamVina* fix_param, FlexParamVina* flex_param,
    FlexPose* x, FlexTopo* flex_topo, FlexPose* out_x_new, FlexPoseGradient* out_g_new,
    Real r = 1.1){
    // set fix
    fix_mol->natom = 1;
    fix_mol->coords = new Real[3];
    fix_mol->coords[0] = 0.;
    fix_mol->coords[1] = 0.;
    fix_mol->coords[2] = 0.;

    fix_param->atom_types = new int[1];
    fix_param->atom_types[0] = VN_TYPE_C_H;

    // set flex
    flex_topo->natom = 2;
    flex_topo->vn_types = new int[2];
    flex_topo->vn_types[0] = VN_TYPE_C_H;
    flex_topo->vn_types[1] = VN_TYPE_C_H;

    prepare_param_by_topo(flex_topo, fix_mol, fix_param, flex_param);

    x->energy = 999;
    x->center[0] = r * 1.5;
    x->center[1] = 0.;
    x->center[2] = 0.;
    x->coords = new Real[6];
    x->coords[0] = r;
    x->coords[1] = 0.;
    x->coords[2] = 0.;
    x->coords[3] = r * 2.;
    x->coords[4] = 0.;
    x->coords[5] = 0.;
    x->dihedrals = nullptr;

    out_x_new->energy = 999;
    out_x_new->coords = new Real[6];
    std::fill(out_x_new->coords, out_x_new->coords + 6, 0.);
    out_x_new->dihedrals = nullptr;

    out_g_new->dihedrals_g = nullptr;
}

TEST_CASE("test cal_e_grad_one", "[cal_e_grad_one]"){
    Real out_e = 9999, out_alpha = -9999;
    FixMol fix_mol;
    FixParamVina fix_param;
    FlexParamVina flex_param;
    FlexPose x;
    FlexTopo flex_topo;
    FlexPose out_x_new;
    FlexPoseGradient out_g_new;
    Vina vina;
    Real f_ref = 0;
    Real e_ref = 0;

    //--------------- Test 1 -----------------
    // One Carbon as receptor, two Carbons as flex
    Real r = 1.3;
    // Set Receptor
    prepare_flex_2(&fix_mol, &fix_param, &flex_param, &x, &flex_topo, &out_x_new, &out_g_new, r);

    cal_e_grad_one_gpu(fix_mol, fix_param, flex_param, x, flex_topo, &out_g_new, &out_e);
    Real e_tmp =0, f_tmp = 0;
    e_tmp = vina.eval_ef(r - flex_param.r1_plus_r2_inter[0], fix_param.atom_types[0], flex_param.atom_types[0], &f_tmp);
    e_ref += e_tmp;
    f_ref += f_tmp;
    e_tmp = vina.eval_ef(2*r - flex_param.r1_plus_r2_inter[1], fix_param.atom_types[0], flex_param.atom_types[1], &f_tmp);
    e_ref += e_tmp;
    f_ref += f_tmp;
    e_tmp = vina.eval_ef(r - flex_param.r1_plus_r2_intra[0], flex_param.atom_types[0], flex_param.atom_types[1], &f_tmp);
    e_ref += e_tmp;

    REQUIRE_THAT(out_e, Catch::Matchers::WithinAbs(e_ref, 1e-4));
    REQUIRE_THAT(out_g_new.center_g[0], Catch::Matchers::WithinAbs(-f_ref, 1e-4));
    REQUIRE_THAT(out_g_new.center_g[1], Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(out_g_new.center_g[2], Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(out_g_new.orientation_g[0], Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(out_g_new.orientation_g[1], Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(out_g_new.orientation_g[2], Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(out_g_new.orientation_g[3], Catch::Matchers::WithinAbs(0, 1e-4));
    //![cal_e_grad_one]
}