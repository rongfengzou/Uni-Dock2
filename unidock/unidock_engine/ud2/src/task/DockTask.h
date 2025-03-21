//
// Created by Congcong Liu on 24-9-23.
//

#ifndef DOCKTASK_H
#define DOCKTASK_H


#include "model/model.h"
#include <string>
#include "score/vina.h"


/**
 * @brief Docking calculation model. Aiming to perform a docking computation method, thus
 * managing all related concepts, including the structure to compute on; the method to
 * apply, results to show...
 */
class DockTask{
public:
    // -------------------- Settings
    // Both - Parameters
    DockParam dock_param;
    int nflex = 0;
    bool show_score = true;

    // -------------------- Molecules
    // CPU - Input Model
    UDFixMol udfix_mol;
    UDFlexMolList udflex_mols;
    std::vector<std::string> fns_flex;

    // CPU - Output Model
    std::string fp_json;

    // Construction
    DockTask(const UDFixMol& fix_mol, const UDFlexMolList& flex_mol_list, DockParam dock_param,
             std::vector<std::string> fns_flex, std::string fp_json):
        udfix_mol(fix_mol), udflex_mols(flex_mol_list), dock_param(dock_param), fns_flex(fns_flex), fp_json(fp_json){
        nflex = flex_mol_list.size();
    };

    /**
     * @brief Run a whole process: global search, cluster by RMSD, refinement by optimization and final output.
     */
    void run();

    // Create dpfix_mol and dpflex_mols
    void from_json(std::string fp); // todo: resume a task from file

    // Output
    void dump_poses_to_json(std::string fp_json);

private:
    // CPU
    int n_atom_all_flex = 0;
    int n_dihe_all_flex = 0;
    std::vector<int> list_i_real; // record indices/ranges of coords & dihedrals in Real data, size: nflex * exhaustiveness
    FlexPose* flex_pose_list_res; // size: nflex * exhaustiveness todo: add a member score instead of these usurpers
    // Finally, use center[3] to record intra, inter, penalty
    // then use orientation[4] to record Predicted Free Energy of Binding, Total score, inter(contains penalty) score, conf_independent part

    Real* flex_pose_list_real_res;
    std::vector<std::vector<int>> clustered_pose_inds_list; // global ind
    std::vector<std::vector<int>> filtered_pose_inds_list; // global ind

    // GPU
    FixMol* fix_mol_cu;
    Real* fix_mol_real_cu;

    FlexPose* flex_pose_list_cu; // size: nflex * exhaustiveness
    Real* flex_pose_list_real_cu;

    FlexTopo* flex_topo_list_cu; // size: nflex
    int* flex_topo_list_int_cu;
    Real* flex_topo_list_real_cu;

    FlexPoseGradient* flex_grad_list_cu;
    FlexPoseHessian* flex_hessian_list_cu;

    FlexParamVina* flex_param_list_cu;
    int* flex_param_list_int_cu;
    Real* flex_param_list_real_cu;
    FixParamVina* fix_param_cu;
    int* fix_param_int_cu;

    Real* aux_list_e_cu; // saves energy of all poses of all flexes, size: nflex * exhaustiveness
    int npose_clustered = 0;
    int* clustered_pose_inds_cu; // pose indices after clustering, -1 indicates that no pose is selected

    // GPU - auxiliary
    FlexPose* aux_poses_cu;
    Real* aux_poses_real_cu;
    FlexPoseGradient* aux_grads_cu;
    Real* aux_grads_real_cu;
    FlexPoseHessian* aux_hessians_cu;
    Real* aux_hessians_real_cu;
    FlexForce* aux_forces_cu;
    Real* aux_forces_real_cu;

    int* aux_rmsd_ij_cu; // saves pose indices of tri mat (no diagonal), {(i, j) | 0<= i,j < exhaustiveness; i < j}
    int* aux_list_cluster_cu; // saves all pose state: 1 for left, 0 for abandoned (after merging)


    // -------------------- Functions
    void prepare_vina();
    void run_search();
    void run_cluster();
    void run_refine();
    void run_filter();
    void run_score();
    void dump_poses();

    void alloc_gpu();
    void cp_to_cpu();
    void free_gpu();
    void free_gpu_mc();
};


#endif //DOCKTASK_H
