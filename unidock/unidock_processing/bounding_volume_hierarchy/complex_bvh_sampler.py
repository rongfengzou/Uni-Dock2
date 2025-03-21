import os
import sys
sys.path.append('/data/aidd_server/Modules/rpxdock')

import itertools
from copy import deepcopy

import numpy as np

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

from OCP.gp import gp_Pnt, gp_Dir

import sampling as sp

from unidock.unidock_processing.ifp_calculation.utils import get_close_ligand_protein_residue_mol_list
from unidock.unidock_processing.torsion_library.utils import rotate_torsion_angle
from unidock.unidock_processing.bounding_volume_hierarchy.utils import check_self_contact_conformation, check_complex_contact_conformation

class ComplexBVHSampler(object):
    def __init__(self,
                 receptor_bvh_builder,
                 ligand_bvh_builder,
                 target_center=(0.0, 0.0, 0.0),
                 box_size=(22.5, 22.5, 22.5),
                 max_torsion_layer_level=(3, 10, 100),
                 translation_division_level=(2, 8, 16),
                 working_dir_name='.'):

        self.receptor_bvh_tree = receptor_bvh_builder.receptor_chain_tree
        self.protein_mol = receptor_bvh_builder.protein_mol
        self.protein_residue_mol_list = receptor_bvh_builder.protein_residue_mol_list

        self.ligand_bvh_tree = ligand_bvh_builder.ligand_torsion_tree
        self.ligand_mol = ligand_bvh_builder.mol
        self.num_ligand_atoms = self.ligand_mol.GetNumAtoms()
        self.ligand_root_atom_idx_list = ligand_bvh_builder.root_atom_idx_list
        self.ligand_center_of_mass = ligand_bvh_builder.center_of_mass
        self.ligand_torsion_layer_array = np.array(ligand_bvh_builder.torsion_layer_list, dtype=np.int32)
        self.num_torsions = ligand_bvh_builder.num_rotatable_bonds
        self.torsion_atom_idx_nested_list = ligand_bvh_builder.torsion_library_driver_bvh.torsion_atom_idx_nested_list
        self.torsion_mobile_atom_idx_nested_list = ligand_bvh_builder.torsion_library_driver_bvh.mobile_atom_idx_nested_list
        self.torsion_library_value_nested_list = ligand_bvh_builder.torsion_library_driver_bvh.enumerated_torsion_value_nested_list

        self.target_center = np.array(target_center, dtype=np.float32)
        self.box_size = np.array(box_size, dtype=np.float32)

        self.max_torsion_layer_level = max_torsion_layer_level
        self.translation_division_level = translation_division_level

        self.working_dir_name = os.path.abspath(working_dir_name)

        rotation_sampler = sp.OriHier_f4(180.0)
        self.rotation_matrix_array = rotation_sampler.get_ori(0, np.arange(rotation_sampler.size(0), dtype=np.int32))[1]
        self.num_orientations = self.rotation_matrix_array.shape[0]

        #################################################################################################################
        #################################################################################################################
        ## Initialize receptor node and edge info
        num_receptor_nodes = len(self.receptor_bvh_tree.nodes)
        self.receptor_node_info_list = [None] * num_receptor_nodes
        self.receptor_edge_info_nested_list = [None] * num_receptor_nodes

        for node_idx in range(num_receptor_nodes):
            node_info_dict = self.receptor_bvh_tree.nodes[node_idx]
            edge_info_list = list(self.receptor_bvh_tree.edges(node_idx))

            self.receptor_node_info_list[node_idx] = node_info_dict
            self.receptor_edge_info_nested_list[node_idx] = edge_info_list

        self.receptor_primary_edge_info_list = self.receptor_edge_info_nested_list[0]
        self.num_receptor_primary_edges = len(self.receptor_primary_edge_info_list)
        #################################################################################################################
        #################################################################################################################

        #################################################################################################################
        #################################################################################################################
        ## Initialize ligand node and edge info
        self.num_ligand_nodes = len(self.ligand_bvh_tree.nodes)
        self.ligand_node_info_list = [None] * self.num_ligand_nodes
        self.ligand_edge_info_nested_list = [None] * self.num_ligand_nodes

        for node_idx in range(self.num_ligand_nodes):
            node_info_dict = self.ligand_bvh_tree.nodes[node_idx]
            edge_info_list = list(self.ligand_bvh_tree.edges(node_idx))

            self.ligand_node_info_list[node_idx] = node_info_dict
            self.ligand_edge_info_nested_list[node_idx] = edge_info_list
        #################################################################################################################
        #################################################################################################################

        #################################################################################################################
        #################################################################################################################
        ## Collect ligand crystal structure torsion values
        self.ligand_crystal_torsion_value_list = [None] * self.num_torsions
        crystal_conformer = self.ligand_mol.GetConformer()

        for torsion_idx in range(self.num_torsions):
            torsion_atom_idx_list = self.torsion_atom_idx_nested_list[torsion_idx]
            self.ligand_crystal_torsion_value_list[torsion_idx] = Chem.rdMolTransforms.GetDihedralDeg(crystal_conformer, *torsion_atom_idx_list)
        #################################################################################################################
        #################################################################################################################

        #################################################################################################################
        #################################################################################################################
        ## Get closed protein residue positions
        closed_protein_residue_mol_list = get_close_ligand_protein_residue_mol_list(self.protein_mol,
                                                                                    self.protein_residue_mol_list,
                                                                                    self.ligand_mol,
                                                                                    ligand_closed_neighbor_cutoff=6.5)

        num_closed_residues = len(closed_protein_residue_mol_list)
        closed_residue_positions_list = [None] * num_closed_residues

        for residue_idx in range(num_closed_residues):
            closed_protein_residue_mol = closed_protein_residue_mol_list[residue_idx]
            closed_residue_positions_list[residue_idx] = closed_protein_residue_mol.GetConformer().GetPositions()

        self.closed_protein_residue_positions = np.vstack(closed_residue_positions_list)
        #################################################################################################################
        #################################################################################################################

    def __single_ligand_node_collision_detection_on_receptor_BVH__(self, ligand_fragment_node):
        ligand_fragment_obb_list = ligand_fragment_node['obb_list']
        ligand_fragment_trimmed_terminal = ligand_fragment_node['trimmed_terminal']

        if ligand_fragment_trimmed_terminal:
            return True

        collision_flag = False

        for ligand_fragment_obb in ligand_fragment_obb_list:
            for primary_edge_idx in range(self.num_receptor_primary_edges):
                primary_edge_info = self.receptor_primary_edge_info_list[primary_edge_idx]
                primary_node_idx = primary_edge_info[1]
                primary_node = self.receptor_node_info_list[primary_node_idx]
                primary_node_obb = primary_node['obb']

                if not ligand_fragment_obb.IsOut(primary_node_obb):
                    collision_flag = True
                    break
                else:
                    secondary_edge_info_list = self.receptor_edge_info_nested_list[primary_node_idx]
                    num_secondary_edges = len(secondary_edge_info_list)
                    if num_secondary_edges > 1:
                        for secondary_edge_idx in range(num_secondary_edges):
                            secondary_edge_info = secondary_edge_info_list[secondary_edge_idx]
                            secondary_node_idx = secondary_edge_info[1]
                            if secondary_node_idx != 0:
                                secondary_node = self.receptor_node_info_list[secondary_node_idx]
                                secondary_node_obb = secondary_node['obb']

                                if not ligand_fragment_obb.IsOut(secondary_node_obb):
                                    collision_flag = True
                                    break

                if collision_flag:
                    break

            if collision_flag:
                break

        return collision_flag

    def __ligand_torsion_tree_collision_detection_on_receptor_BVH__(self, ligand_current_node_idx, ligand_parent_node_idx, max_torsion_layer):
        node = self.ligand_node_info_list[ligand_current_node_idx]
        torsion_layer = node['torsion_layer']

        collision_status = self.__single_ligand_node_collision_detection_on_receptor_BVH__(node)
        if not collision_status:
            children_node_idx_list = []
            children_edge_info_list = self.ligand_edge_info_nested_list[ligand_current_node_idx]
            num_possible_children_edges = len(children_edge_info_list)
            for children_edge_idx in range(num_possible_children_edges):
                children_edge_info = children_edge_info_list[children_edge_idx]
                children_node_idx = children_edge_info[1]

                if ligand_parent_node_idx is not None and children_node_idx == ligand_parent_node_idx:
                    continue
                else:
                    children_node_idx_list.append(children_node_idx)

            num_children_nodes = len(children_node_idx_list)

            if num_children_nodes == 0 or torsion_layer == max_torsion_layer:
                torsion_value_list = node['torsion_value_list']
                self.temp_accepted_torsion_value_nested_list.append(torsion_value_list)
            else:
                for children_idx in range(num_children_nodes):
                    children_node_idx = children_node_idx_list[children_idx]
                    self.__ligand_torsion_tree_collision_detection_on_receptor_BVH__(children_node_idx, ligand_current_node_idx, max_torsion_layer)

    def __translate_rotate_ligand_BVH__(self, translated_center_of_mass, rotation_matrix, max_torsion_layer):
        for node_idx in range(self.num_ligand_nodes):
            node = self.ligand_node_info_list[node_idx]

            torsion_layer = node['torsion_layer']
            if torsion_layer > max_torsion_layer:
                continue

            obb_list = node['obb_list']
            obb_info_dict_list = node['obb_info_dict_list']

            num_obbs = len(obb_list)
            for obb_idx in range(num_obbs):
                obb = obb_list[obb_idx]
                obb_info_dict = obb_info_dict_list[obb_idx]

                center = obb_info_dict['center']
                x_axis_vector = obb_info_dict['x_axis_vector']
                y_axis_vector = obb_info_dict['y_axis_vector']
                z_axis_vector = obb_info_dict['z_axis_vector']
                x_axis_size = obb_info_dict['x_axis_size']
                y_axis_size = obb_info_dict['y_axis_size']
                z_axis_size = obb_info_dict['z_axis_size']

                center_coords = center - self.ligand_center_of_mass
                x_coords = x_axis_vector + center_coords
                y_coords = y_axis_vector + center_coords
                z_coords = z_axis_vector + center_coords

                rotated_center = np.matmul(rotation_matrix, center_coords.T).T
                rotated_x_coords = np.matmul(rotation_matrix, x_coords.T).T
                rotated_y_coords = np.matmul(rotation_matrix, y_coords.T).T
                rotated_z_coords = np.matmul(rotation_matrix, z_coords.T).T

                transformed_center = rotated_center + translated_center_of_mass
                transformed_x_axis = rotated_x_coords - rotated_center
                transformed_y_axis = rotated_y_coords - rotated_center
                transformed_z_axis = rotated_z_coords - rotated_center

                transformed_center = gp_Pnt(*transformed_center)
                transformed_x_axis = gp_Dir(*transformed_x_axis)
                transformed_y_axis = gp_Dir(*transformed_y_axis)
                transformed_z_axis = gp_Dir(*transformed_z_axis)

                obb.SetCenter(transformed_center)
                obb.SetXComponent(transformed_x_axis, x_axis_size)
                obb.SetYComponent(transformed_y_axis, y_axis_size)
                obb.SetZComponent(transformed_z_axis, z_axis_size)

    def __check_collision_configurations__(self, torsion_value_nested_list, max_torsion_layer):
        checked_torsion_idx_array = np.where(self.ligand_torsion_layer_array <= max_torsion_layer)[0]
        for torsion_idx in checked_torsion_idx_array:
            exist_torsion_conformation = False
            for torsion_value_list in torsion_value_nested_list:
                torsion_value = torsion_value_list[torsion_idx]
                if torsion_value is not None:
                    exist_torsion_conformation = True
                    break

            if not exist_torsion_conformation:
                return False
    
        return True

    def __check_conflict_torsion_path__(self, torsion_value_nested_list):
        num_torsions = len(torsion_value_nested_list[0])
        torsion_path_info_dict = {}
        for torsion_value_list in torsion_value_nested_list:
            torsion_value_path_list = []
            torsion_idx_path_list = []
            for torsion_idx in range(num_torsions):
                torsion_value = torsion_value_list[torsion_idx]
                if torsion_value is not None:
                    torsion_value_path_list.append(torsion_value)
                    torsion_idx_path_list.append(torsion_idx)

            torsion_idx_path_tuple = tuple(torsion_idx_path_list)
            if torsion_idx_path_tuple not in torsion_path_info_dict:
                torsion_path_info_dict[torsion_idx_path_tuple] = []

            torsion_path_info_dict[torsion_idx_path_tuple].append(torsion_value_path_list)

        torsion_idx_path_tuple_list = list(torsion_path_info_dict.keys())
        num_torsion_paths = len(torsion_idx_path_tuple_list)
        for current_torsion_path_idx in range(num_torsion_paths):
            current_torsion_idx_path_tuple = torsion_idx_path_tuple_list[current_torsion_path_idx]
            current_torsion_idx_array = np.array(current_torsion_idx_path_tuple, dtype=np.int32)
            current_torsion_sample_nested_list = torsion_path_info_dict[current_torsion_idx_path_tuple]
            current_torsion_sample_array = np.array(current_torsion_sample_nested_list, dtype=np.float32)
            if current_torsion_sample_array.shape[0] == 0:
                continue

            for compared_torsion_path_idx in range(num_torsion_paths):
                if compared_torsion_path_idx == current_torsion_path_idx:
                    continue

                compared_torsion_idx_path_tuple = torsion_idx_path_tuple_list[compared_torsion_path_idx]
                compared_torsion_idx_array = np.array(compared_torsion_idx_path_tuple, dtype=np.int32)
                compared_torsion_sample_nested_list = torsion_path_info_dict[compared_torsion_idx_path_tuple]
                compared_torsion_sample_array = np.array(compared_torsion_sample_nested_list, dtype=np.float32)

                if len(compared_torsion_sample_nested_list) == 0:
                    continue

                common_torsion_idx_list = list(set(current_torsion_idx_array).intersection(set(compared_torsion_idx_array)))
                common_torsion_idx_array = np.array(common_torsion_idx_list, dtype=np.int32)
                if common_torsion_idx_array.shape[0] == 0:
                    continue

                current_common_idx_array = np.where(np.isin(current_torsion_idx_array, common_torsion_idx_array))[0]
                compared_common_idx_array = np.where(np.isin(compared_torsion_idx_array, common_torsion_idx_array))[0]

                current_common_torsion_sample_array = current_torsion_sample_array[:, current_common_idx_array]
                unique_common_torsion_sample_nested_list = np.unique(current_common_torsion_sample_array, axis=0).tolist()

                num_compared_torsion_samples = compared_torsion_sample_array.shape[0]
                compared_common_torsion_sample_array = compared_torsion_sample_array[:, compared_common_idx_array]

                removed_compared_torsion_sample_nested_list = []
                for compared_idx in range(num_compared_torsion_samples):
                    if compared_common_torsion_sample_array[compared_idx, :].tolist() not in unique_common_torsion_sample_nested_list:
                        compared_torsion_sample_list = compared_torsion_sample_nested_list[compared_idx]
                        removed_compared_torsion_sample_nested_list.append(compared_torsion_sample_list)

                for compared_torsion_sample_list in removed_compared_torsion_sample_nested_list:
                    compared_torsion_sample_nested_list.remove(compared_torsion_sample_list)

        no_conflict_configuration = True
        for torsion_path_idx in range(num_torsion_paths):
            torsion_idx_path_tuple = torsion_idx_path_tuple_list[torsion_path_idx]
            if len(torsion_path_info_dict[torsion_idx_path_tuple]) == 0:
                no_conflict_configuration = False
                break

        return torsion_path_info_dict, no_conflict_configuration

    def __enumerate_possible_torsion_conformations__(self, torsion_path_info_dict):
        torsion_idx_path_tuple_list = list(torsion_path_info_dict.keys())
        num_torsion_paths = len(torsion_idx_path_tuple_list)
        merged_torsion_idx_path_tuple = torsion_idx_path_tuple_list[0]
        merged_torsion_idx_array = np.array(merged_torsion_idx_path_tuple, dtype=np.int32)
        merged_torsion_sample_nested_list = torsion_path_info_dict[merged_torsion_idx_path_tuple]
        merged_torsion_sample_array = np.array(merged_torsion_sample_nested_list, dtype=np.float32)

        for torsion_path_idx in range(1, num_torsion_paths):
            torsion_idx_path_tuple = torsion_idx_path_tuple_list[torsion_path_idx]
            torsion_idx_array = np.array(torsion_idx_path_tuple, dtype=np.int32)
            torsion_sample_nested_list = torsion_path_info_dict[torsion_idx_path_tuple]
            torsion_sample_array = np.array(torsion_sample_nested_list, dtype=np.float32)

            common_torsion_idx_array, merged_common_idx_array, current_common_idx_array = np.intersect1d(merged_torsion_idx_array,
                                                                                                         torsion_idx_array,
                                                                                                         assume_unique=True,
                                                                                                         return_indices=True)

            merged_diff_torsion_idx_array = np.setdiff1d(merged_torsion_idx_array, torsion_idx_array, assume_unique=True)
            current_diff_torsion_idx_array = np.setdiff1d(torsion_idx_array, merged_torsion_idx_array, assume_unique=True)

            merged_diff_idx_array = np.where(np.isin(merged_torsion_idx_array, merged_diff_torsion_idx_array))[0]
            current_diff_idx_array = np.where(np.isin(torsion_idx_array, current_diff_torsion_idx_array))[0]

            if common_torsion_idx_array.shape[0] == 0:
                merged_torsion_idx_path_tuple = merged_torsion_idx_path_tuple + torsion_idx_path_tuple
                merged_torsion_idx_array = np.array(merged_torsion_idx_path_tuple, dtype=np.int32)

                enumerated_torsion_sample_set_list = list(itertools.product(merged_torsion_sample_nested_list, torsion_sample_nested_list, repeat=1))
                num_merged_samples = len(enumerated_torsion_sample_set_list)
                merged_torsion_sample_nested_list = [None] * num_merged_samples
                for merged_sample_idx in range(num_merged_samples):
                    enumerated_torsion_sample_set = enumerated_torsion_sample_set_list[merged_sample_idx]
                    merged_torsion_sample_list = [*enumerated_torsion_sample_set[0], *enumerated_torsion_sample_set[1]]
                    merged_torsion_sample_nested_list[merged_sample_idx] = merged_torsion_sample_list

                merged_torsion_sample_array = np.array(merged_torsion_sample_nested_list, dtype=np.float32)

            else:
                merged_torsion_idx_path_tuple = tuple(np.hstack([common_torsion_idx_array, merged_diff_torsion_idx_array, current_diff_torsion_idx_array]))
                merged_torsion_idx_array = np.array(merged_torsion_idx_path_tuple, dtype=np.int32)

                merged_torsion_sample_nested_list = []

                for merged_torsion_sample in merged_torsion_sample_array:
                    merged_common_torsion_sample = merged_torsion_sample[merged_common_idx_array]
                    merged_diff_torsion_sample = merged_torsion_sample[merged_diff_idx_array]

                    for torsion_sample in torsion_sample_array:
                        common_torsion_sample = torsion_sample[current_common_idx_array]
                        diff_torsion_sample = torsion_sample[current_diff_idx_array]

                        if np.allclose(merged_common_torsion_sample, common_torsion_sample):
                            merged_torsion_sample_list = np.hstack([merged_common_torsion_sample, merged_diff_torsion_sample, diff_torsion_sample]).tolist()
                            merged_torsion_sample_nested_list.append(merged_torsion_sample_list)

                merged_torsion_sample_array = np.array(merged_torsion_sample_nested_list, dtype=np.float32)

        sorted_idx_array = np.argsort(merged_torsion_idx_array)
        sorted_merged_torsion_idx_array = merged_torsion_idx_array[sorted_idx_array]
        sorted_merged_torsion_sample_array = merged_torsion_sample_array[:, sorted_idx_array]

        #################################################################################################
        #################################################################################################
        ## Enumerate unsampled torsions
        total_torsion_idx_array = np.arange(self.num_torsions, dtype=np.int32)
        unsampled_torsion_idx_array = np.setdiff1d(total_torsion_idx_array, sorted_merged_torsion_idx_array, assume_unique=True)
        num_unsampled_torsions = unsampled_torsion_idx_array.shape[0]
        if num_unsampled_torsions > 0:
            merged_torsion_idx_array = sorted_merged_torsion_idx_array
            merged_torsion_sample_nested_list = sorted_merged_torsion_sample_array.tolist()
            for unsampled_idx in range(num_unsampled_torsions):
                unsampled_torsion_idx = unsampled_torsion_idx_array[unsampled_idx]
                merged_torsion_idx_array = np.hstack([merged_torsion_idx_array, np.array([unsampled_torsion_idx], dtype=np.int32)])
                unsampled_torsion_value_array = np.array(self.torsion_library_value_nested_list[unsampled_torsion_idx], np.float32)
                num_posibilities = unsampled_torsion_value_array.shape[0]
                unsampled_torsion_value_reshaped_list = np.reshape(unsampled_torsion_value_array, (num_posibilities, 1)).tolist()
                enumerated_torsion_sample_set_list = list(itertools.product(merged_torsion_sample_nested_list, unsampled_torsion_value_reshaped_list, repeat=1))

                num_merged_samples = len(enumerated_torsion_sample_set_list)
                merged_torsion_sample_nested_list = [None] * num_merged_samples
                for merged_sample_idx in range(num_merged_samples):
                    enumerated_torsion_sample_set = enumerated_torsion_sample_set_list[merged_sample_idx]
                    merged_torsion_sample_list = [*enumerated_torsion_sample_set[0], *enumerated_torsion_sample_set[1]]
                    merged_torsion_sample_nested_list[merged_sample_idx] = merged_torsion_sample_list

            merged_torsion_sample_array = np.array(merged_torsion_sample_nested_list, dtype=np.float32)

            sorted_idx_array = np.argsort(merged_torsion_idx_array)
            sorted_merged_torsion_idx_array = merged_torsion_idx_array[sorted_idx_array]
            sorted_merged_torsion_sample_array = merged_torsion_sample_array[:, sorted_idx_array]
        #################################################################################################
        #################################################################################################

        return sorted_merged_torsion_sample_array

    def __enumerate_possible_torsion_conformations_crystal__(self, torsion_path_info_dict):
        torsion_idx_path_tuple_list = list(torsion_path_info_dict.keys())
        num_torsion_paths = len(torsion_idx_path_tuple_list)
        merged_torsion_idx_path_tuple = torsion_idx_path_tuple_list[0]
        merged_torsion_idx_array = np.array(merged_torsion_idx_path_tuple, dtype=np.int32)
        merged_torsion_sample_nested_list = torsion_path_info_dict[merged_torsion_idx_path_tuple]
        merged_torsion_sample_array = np.array(merged_torsion_sample_nested_list, dtype=np.float32)

        for torsion_path_idx in range(1, num_torsion_paths):
            torsion_idx_path_tuple = torsion_idx_path_tuple_list[torsion_path_idx]
            torsion_idx_array = np.array(torsion_idx_path_tuple, dtype=np.int32)
            torsion_sample_nested_list = torsion_path_info_dict[torsion_idx_path_tuple]
            torsion_sample_array = np.array(torsion_sample_nested_list, dtype=np.float32)

            common_torsion_idx_array, merged_common_idx_array, current_common_idx_array = np.intersect1d(merged_torsion_idx_array,
                                                                                                         torsion_idx_array,
                                                                                                         assume_unique=True,
                                                                                                         return_indices=True)

            merged_diff_torsion_idx_array = np.setdiff1d(merged_torsion_idx_array, torsion_idx_array, assume_unique=True)
            current_diff_torsion_idx_array = np.setdiff1d(torsion_idx_array, merged_torsion_idx_array, assume_unique=True)

            merged_diff_idx_array = np.where(np.isin(merged_torsion_idx_array, merged_diff_torsion_idx_array))[0]
            current_diff_idx_array = np.where(np.isin(torsion_idx_array, current_diff_torsion_idx_array))[0]

            if common_torsion_idx_array.shape[0] == 0:
                merged_torsion_idx_path_tuple = merged_torsion_idx_path_tuple + torsion_idx_path_tuple
                merged_torsion_idx_array = np.array(merged_torsion_idx_path_tuple, dtype=np.int32)

                enumerated_torsion_sample_set_list = list(itertools.product(merged_torsion_sample_nested_list, torsion_sample_nested_list, repeat=1))
                num_merged_samples = len(enumerated_torsion_sample_set_list)
                merged_torsion_sample_nested_list = [None] * num_merged_samples
                for merged_sample_idx in range(num_merged_samples):
                    enumerated_torsion_sample_set = enumerated_torsion_sample_set_list[merged_sample_idx]
                    merged_torsion_sample_list = [*enumerated_torsion_sample_set[0], *enumerated_torsion_sample_set[1]]
                    merged_torsion_sample_nested_list[merged_sample_idx] = merged_torsion_sample_list

                merged_torsion_sample_array = np.array(merged_torsion_sample_nested_list, dtype=np.float32)

            else:
                merged_torsion_idx_path_tuple = tuple(np.hstack([common_torsion_idx_array, merged_diff_torsion_idx_array, current_diff_torsion_idx_array]))
                merged_torsion_idx_array = np.array(merged_torsion_idx_path_tuple, dtype=np.int32)

                merged_torsion_sample_nested_list = []

                for merged_torsion_sample in merged_torsion_sample_array:
                    merged_common_torsion_sample = merged_torsion_sample[merged_common_idx_array]
                    merged_diff_torsion_sample = merged_torsion_sample[merged_diff_idx_array]

                    for torsion_sample in torsion_sample_array:
                        common_torsion_sample = torsion_sample[current_common_idx_array]
                        diff_torsion_sample = torsion_sample[current_diff_idx_array]

                        if np.allclose(merged_common_torsion_sample, common_torsion_sample):
                            merged_torsion_sample_list = np.hstack([merged_common_torsion_sample, merged_diff_torsion_sample, diff_torsion_sample]).tolist()
                            merged_torsion_sample_nested_list.append(merged_torsion_sample_list)

                merged_torsion_sample_array = np.array(merged_torsion_sample_nested_list, dtype=np.float32)

        sorted_idx_array = np.argsort(merged_torsion_idx_array)
        sorted_merged_torsion_idx_array = merged_torsion_idx_array[sorted_idx_array]
        sorted_merged_torsion_sample_array = merged_torsion_sample_array[:, sorted_idx_array]

        #################################################################################################
        #################################################################################################
        ## Assign crystal value for unsampled torsions
        total_torsion_idx_array = np.arange(self.num_torsions, dtype=np.int32)
        unsampled_torsion_idx_array = np.setdiff1d(total_torsion_idx_array, sorted_merged_torsion_idx_array, assume_unique=True)
        num_unsampled_torsions = unsampled_torsion_idx_array.shape[0]

        if num_unsampled_torsions > 0:
            merged_torsion_idx_array = np.hstack([sorted_merged_torsion_idx_array, unsampled_torsion_idx_array])
            merged_torsion_sample_nested_list = sorted_merged_torsion_sample_array.tolist()

            num_merged_samples = len(merged_torsion_sample_nested_list)
            crystal_merged_torsion_sample_nested_list = [None] * num_merged_samples

            for merged_sample_idx in range(num_merged_samples):
                merged_torsion_sample_list = merged_torsion_sample_nested_list[merged_sample_idx]
                unsampled_torsion_sample_list = [None] * num_unsampled_torsions

                for unsampled_idx in range(num_unsampled_torsions):
                    unsampled_torsion_idx = unsampled_torsion_idx_array[unsampled_idx]
                    unsampled_torsion_sample_list[unsampled_idx] = self.ligand_crystal_torsion_value_list[unsampled_torsion_idx]

                crystal_merged_torsion_sample_list = merged_torsion_sample_list + unsampled_torsion_sample_list
                crystal_merged_torsion_sample_nested_list[merged_sample_idx] = crystal_merged_torsion_sample_list

            merged_torsion_sample_array = np.array(crystal_merged_torsion_sample_nested_list, dtype=np.float32)

            sorted_idx_array = np.argsort(merged_torsion_idx_array)
            sorted_merged_torsion_idx_array = merged_torsion_idx_array[sorted_idx_array]
            sorted_merged_torsion_sample_array = merged_torsion_sample_array[:, sorted_idx_array]
        #################################################################################################
        #################################################################################################

        return sorted_merged_torsion_sample_array

    def __find_translation_sample_spacing__(self, translated_center_of_mass_array):
        temp_translated_center_of_mass_x = translated_center_of_mass_array[0, 0]
        temp_translated_center_of_mass_y = translated_center_of_mass_array[0, 1]
        temp_translated_center_of_mass_z = translated_center_of_mass_array[0, 2]

        test_translated_center_of_mass_array_x = translated_center_of_mass_array[1:, 0]
        test_translated_center_of_mass_array_y = translated_center_of_mass_array[1:, 1]
        test_translated_center_of_mass_array_z = translated_center_of_mass_array[1:, 2]

        x_distance_array = test_translated_center_of_mass_array_x - temp_translated_center_of_mass_x
        y_distance_array = test_translated_center_of_mass_array_y - temp_translated_center_of_mass_y
        z_distance_array = test_translated_center_of_mass_array_z - temp_translated_center_of_mass_z

        x_distance_array = x_distance_array[np.nonzero(x_distance_array)[0]]
        y_distance_array = y_distance_array[np.nonzero(y_distance_array)[0]]
        z_distance_array = z_distance_array[np.nonzero(z_distance_array)[0]]

        x_sample_spacing = abs(np.min(x_distance_array))
        y_sample_spacing = abs(np.min(y_distance_array))
        z_sample_spacing = abs(np.min(z_distance_array))

        return np.array([x_sample_spacing, y_sample_spacing, z_sample_spacing], dtype=np.float32)

    def __get_tranlation_sample_arrays__(self, sampling_level):
        translation_division = self.translation_division_level[sampling_level]
        translation_spacing = np.array([translation_division, translation_division, translation_division], dtype=np.int32)

        if sampling_level == 0 or len(self.accepted_center_of_mass_list) == 0:
            box_center_x = self.target_center[0]
            box_center_y = self.target_center[1]
            box_center_z = self.target_center[2]

            half_box_size_x = self.box_size[0] / 2.0
            half_box_size_y = self.box_size[1] / 2.0
            half_box_size_z = self.box_size[2] / 2.0

            center_lower_bound_x = box_center_x - half_box_size_x
            center_lower_bound_y = box_center_y - half_box_size_y
            center_lower_bound_z = box_center_z - half_box_size_z

            center_upper_bound_x = box_center_x + half_box_size_x
            center_upper_bound_y = box_center_y + half_box_size_y
            center_upper_bound_z = box_center_z + half_box_size_z

            center_lower_bound_vector = np.array([center_lower_bound_x, center_lower_bound_y, center_lower_bound_z], dtype=np.float32)
            center_upper_bound_vector = np.array([center_upper_bound_x, center_upper_bound_y, center_upper_bound_z], dtype=np.float32)

            translation_spacing = np.array([translation_division, translation_division, translation_division], dtype=np.int32)

            translation_sampler = sp.CartHier3D_f4(center_lower_bound_vector, center_upper_bound_vector, translation_spacing)
            self.translated_center_of_mass_array = translation_sampler.get_trans(0, np.arange(translation_sampler.size(0), dtype=np.int32))[1].astype(np.float32)
            self.translation_sample_spacing = self.__find_translation_sample_spacing__(self.translated_center_of_mass_array)

        else:
            half_box_size_x = self.translation_sample_spacing[0]
            half_box_size_y = self.translation_sample_spacing[1]
            half_box_size_z = self.translation_sample_spacing[2]

            num_previous_accepted_translation_states = len(self.accepted_center_of_mass_list)
            translated_center_of_mass_array_list = [None] * num_previous_accepted_translation_states

            for translation_idx in range(num_previous_accepted_translation_states):
                accepted_center_of_mass = self.accepted_center_of_mass_list[translation_idx]
                box_center_x = accepted_center_of_mass[0]
                box_center_y = accepted_center_of_mass[1]
                box_center_z = accepted_center_of_mass[2]

                center_lower_bound_x = box_center_x - half_box_size_x
                center_lower_bound_y = box_center_y - half_box_size_y
                center_lower_bound_z = box_center_z - half_box_size_z

                center_upper_bound_x = box_center_x + half_box_size_x
                center_upper_bound_y = box_center_y + half_box_size_y
                center_upper_bound_z = box_center_z + half_box_size_z

                center_lower_bound_vector = np.array([center_lower_bound_x, center_lower_bound_y, center_lower_bound_z], dtype=np.float32)
                center_upper_bound_vector = np.array([center_upper_bound_x, center_upper_bound_y, center_upper_bound_z], dtype=np.float32)

                translation_sampler = sp.CartHier3D_f4(center_lower_bound_vector, center_upper_bound_vector, translation_spacing)
                translated_center_of_mass_array_list[translation_idx] = translation_sampler.get_trans(0, np.arange(translation_sampler.size(0), dtype=np.int32))[1]

            self.translation_sample_spacing = self.__find_translation_sample_spacing__(translated_center_of_mass_array_list[0])
            self.translated_center_of_mass_array = np.vstack(translated_center_of_mass_array_list).astype(np.float32)

        self.num_translations = self.translated_center_of_mass_array.shape[0]

    def complex_BVH_static_collision_test(self, max_torsion_layer):
        self.temp_accepted_torsion_value_nested_list = []
        self.__ligand_torsion_tree_collision_detection_on_receptor_BVH__(0, None, max_torsion_layer)

        return self.temp_accepted_torsion_value_nested_list

    def sample_translation_rotation_ligand_BVH_collision_test(self, sampling_level):
        self.__get_tranlation_sample_arrays__(sampling_level)
        max_torsion_layer = self.max_torsion_layer_level[sampling_level]

        self.filtered_torsion_value_nested_list = []
        self.accepted_torsion_value_nested_list = []
        self.sampled_configuration_state_list = []

        accepted_center_of_mass_nested_list = []

        for translation_idx in range(self.num_translations):
            translated_center_of_mass = self.translated_center_of_mass_array[translation_idx, :]

            for orientation_idx in range(self.num_orientations):
                rotation_matrix = self.rotation_matrix_array[orientation_idx, :, :]

                self.__translate_rotate_ligand_BVH__(translated_center_of_mass, rotation_matrix, max_torsion_layer)

                current_accepted_torsion_value_nested_list = self.complex_BVH_static_collision_test(max_torsion_layer)
                configuration_state_tuple = (translated_center_of_mass, rotation_matrix, current_accepted_torsion_value_nested_list)

                if len(current_accepted_torsion_value_nested_list) > 0 and self.__check_collision_configurations__(current_accepted_torsion_value_nested_list, max_torsion_layer):
                    torsion_path_info_dict, no_conflict_configuration = self.__check_conflict_torsion_path__(current_accepted_torsion_value_nested_list)

                    if no_conflict_configuration:
                        self.accepted_torsion_value_nested_list.append(configuration_state_tuple)
                        accepted_center_of_mass_nested_list.append(translated_center_of_mass.tolist())

                        enumerated_torsion_sample_array = self.__enumerate_possible_torsion_conformations_crystal__(torsion_path_info_dict)
                        enumerated_configuration_state_tuple = (translated_center_of_mass, rotation_matrix, enumerated_torsion_sample_array)
                        self.sampled_configuration_state_list.append(enumerated_configuration_state_tuple)

                    else:
                        self.filtered_torsion_value_nested_list.append(configuration_state_tuple)
                else:
                    self.filtered_torsion_value_nested_list.append(configuration_state_tuple)

        accepted_center_of_mass_array = np.array(accepted_center_of_mass_nested_list, dtype=np.float32)
        self.accepted_center_of_mass_list = np.unique(accepted_center_of_mass_array, axis=0).tolist()

    def __generate_sampled_conformations__(self, configuration_state_tuple):
        valid_torsion_state_nested_list = []
        valid_generated_positions_nested_list = []

        translated_center_of_mass = configuration_state_tuple[0]
        rotation_matrix = configuration_state_tuple[1]
        torsion_sample_array = configuration_state_tuple[2]

        root_atom_idx = self.ligand_root_atom_idx_list[0]
        mol = deepcopy(self.ligand_mol)
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        coords = positions - self.ligand_center_of_mass
        transformed_positions = np.matmul(rotation_matrix, coords.T).T + translated_center_of_mass

        for atom_idx in range(self.num_ligand_atoms):
            atom_coords = transformed_positions[atom_idx, :]
            atom_point_3D = Point3D(atom_coords[0], atom_coords[1], atom_coords[2])
            conformer.SetAtomPosition(atom_idx, atom_point_3D)

        for torsion_sample in torsion_sample_array:
            for torsion_idx in range(self.num_torsions):
                torsion_atom_idx_list = self.torsion_atom_idx_nested_list[torsion_idx]
                torsion_mobile_atom_idx_list = self.torsion_mobile_atom_idx_nested_list[torsion_idx]
                torsion_value = torsion_sample[torsion_idx]
                rotate_torsion_angle(mol, torsion_atom_idx_list, torsion_mobile_atom_idx_list, float(torsion_value))

            generated_positions = conformer.GetPositions().astype(np.float32)
            if not check_self_contact_conformation(generated_positions):
                collision_flag, free_flag = check_complex_contact_conformation(self.closed_protein_residue_positions, generated_positions)
                if not collision_flag and not free_flag:
                    valid_torsion_state_nested_list.append(torsion_sample.tolist())
                    valid_generated_positions_nested_list.append(generated_positions.tolist())

        if len(valid_torsion_state_nested_list) > 0:
            valid_generated_positions_array = np.array(valid_generated_positions_nested_list, dtype=np.float32)
            valid_torsion_sample_array = np.array(valid_torsion_state_nested_list, dtype=np.float32)
            valid_configuration_state_tuple = (translated_center_of_mass, rotation_matrix, valid_torsion_sample_array)
        else:
            valid_generated_positions_array = None
            valid_configuration_state_tuple = None

        return valid_generated_positions_array, valid_configuration_state_tuple

    def generate_selected_sampled_valid_conformations(self):
        self.sampled_valid_configuration_state_list = []
        self.sampled_valid_conformations_list = []

        num_translation_rotation_states = len(self.sampled_configuration_state_list)
        selected_configuration_state_list = [None] * num_translation_rotation_states

        for idx in range(1000):
            for state_idx in range(num_translation_rotation_states):
                translation_rotation_state_tuple = self.sampled_configuration_state_list[state_idx]
                torsion_states_array = translation_rotation_state_tuple[2]
                num_current_torsion_states = torsion_states_array.shape[0]
                random_torsion_state_idx = np.random.choice(num_current_torsion_states)
                selected_torsion_state = np.array([torsion_states_array[random_torsion_state_idx, :].tolist()], dtype=np.float32)

                selected_configuration_state_list[state_idx] = (translation_rotation_state_tuple[0],
                                                                translation_rotation_state_tuple[1],
                                                                selected_torsion_state)

            for sampled_configuration_state_tuple in selected_configuration_state_list:
                valid_generated_positions_array, valid_configuration_state_tuple = self.__generate_sampled_conformations__(sampled_configuration_state_tuple)

                if valid_generated_positions_array is not None:
                    self.sampled_valid_configuration_state_list.append(valid_configuration_state_tuple)
                    self.sampled_valid_conformations_list.append(valid_generated_positions_array)

            if len(self.sampled_valid_conformations_list) > 1000:
                break

    def write_sampled_conformations(self):
        conf_idx = 0
        num_transformed_states = len(self.sampled_valid_conformations_list)

        for transformed_state_idx in range(num_transformed_states):
            generated_positions_array = self.sampled_valid_conformations_list[transformed_state_idx]
            num_conformations = generated_positions_array.shape[0]

            for generated_conf_idx in range(num_conformations):
                current_positions_array = generated_positions_array[generated_conf_idx, :, :]

                mol = deepcopy(self.ligand_mol)
                conformer = mol.GetConformer()

                for atom_idx in range(self.num_ligand_atoms):
                    atom_coords = current_positions_array[atom_idx, :]
                    atom_point_3D = Point3D(float(atom_coords[0]), float(atom_coords[1]), float(atom_coords[2]))
                    conformer.SetAtomPosition(atom_idx, atom_point_3D)

                sampled_conformation_sdf_file_name = os.path.join(self.working_dir_name, f'sampled_conformation_{conf_idx}.sdf')
                writer = Chem.SDWriter(sampled_conformation_sdf_file_name)
                writer.write(mol)
                writer.flush()
                writer.close()
                conf_idx += 1

        self.num_sampled_conformations = conf_idx
