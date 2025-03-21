import os
import warnings
from copy import deepcopy
import dill as pickle

import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdDepictor import Compute2DCoords

from unidock.unidock_processing.ligand_topology.generic_rotatable_bond import GenericRotatableBond
from unidock.unidock_processing.ligand_topology.utils import calculate_center_of_mass, assign_atom_properties
from unidock.unidock_processing.torsion_library.torsion_library_driver import TorsionLibraryDriver
from unidock.unidock_processing.bounding_volume_hierarchy.utils import construct_oriented_bounding_box_list, mol2image, create_network_view

class LigandBVHBuilder(object):
    def __init__(self,
                 ligand_sdf_file_name,
                 create_tree_visualization=True):

        self.ligand_sdf_file_name = os.path.abspath(ligand_sdf_file_name)
        self.mol = Chem.SDMolSupplier(self.ligand_sdf_file_name, removeHs=False)[0]
        ComputeGasteigerCharges(self.mol)
        assign_atom_properties(self.mol)
        self.center_of_mass = calculate_center_of_mass(self.mol)

        torsion_library_pkl_file_name = os.path.join(os.path.dirname(__file__), '..', 'torsion_library', 'data', 'torsion_library.pkl')
        with open(torsion_library_pkl_file_name, 'rb') as torsion_library_pkl_file:
            self.torsion_library_dict = pickle.load(torsion_library_pkl_file)

        self.rotatable_bond_finder = GenericRotatableBond()
        self.rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(self.mol)
        self.num_rotatable_bonds = len(self.rotatable_bond_info_list)
        self.torsion_layer_list = [None] * self.num_rotatable_bonds
        self.torsion_library_driver_bvh = TorsionLibraryDriver(self.mol, self.rotatable_bond_info_list, self.torsion_library_dict)
        self.torsion_library_driver_bvh.perform_torsion_matches()

        self.create_tree_visualization = create_tree_visualization

    def build_ligand_BVH_tree(self):
        ##############################################################################
        ##############################################################################
        ## Build Fragments for both OBB and visualizations
        self.mol_visualized = deepcopy(self.mol)
        self.mol_visualized.RemoveAllConformers()
        Compute2DCoords(self.mol_visualized)

        bond_list = list(self.mol.GetBonds())
        rotatable_bond_idx_list = []
        for bond in bond_list:
            bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            if bond_info in self.rotatable_bond_info_list or bond_info_reversed in self.rotatable_bond_info_list:
                rotatable_bond_idx_list.append(bond.GetIdx())

        if len(rotatable_bond_idx_list) > 0:
            splitted_mol = FragmentOnBonds(self.mol, rotatable_bond_idx_list, addDummies=False)
            splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))

            splitted_mol_visualized = FragmentOnBonds(self.mol_visualized, rotatable_bond_idx_list, addDummies=False)
            splitted_mol_list_visualized = list(GetMolFrags(splitted_mol_visualized, asMols=True, sanitizeFrags=False))

        else:
            splitted_mol_list = [self.mol]
            splitted_mol_list_visualized = [self.mol_visualized]

        self.num_fragments = len(splitted_mol_list)
        self.fragment_mol_list = splitted_mol_list
        self.fragment_mol_list_visualized = splitted_mol_list_visualized

        for fragment_mol in self.fragment_mol_list:
            Chem.GetSymmSSSR(fragment_mol)
            fragment_mol.UpdatePropertyCache(strict=False)
        ##############################################################################
        ##############################################################################

        ## Find fragment as the root node
        ##############################################################################
        num_fragment_atoms_list = [None] * self.num_fragments
        for fragment_idx in range(self.num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        root_fragment_idx = np.argmax(num_fragment_atoms_list)
        ##############################################################################

        ## Build torsion tree
        ### Build nodes
        ##############################################################################
        self.ligand_torsion_tree = nx.Graph()
        self.ligand_torsion_tree_visualized = nx.Graph()
        self.current_node_idx = 0

        root_fragment = self.fragment_mol_list[root_fragment_idx]
        root_fragment_visualized = self.fragment_mol_list_visualized[root_fragment_idx]
        num_root_atoms = root_fragment.GetNumAtoms()
        root_fragment_atom_idx_list = [atom.GetIntProp('internal_atom_idx') for atom in root_fragment.GetAtoms()]
        self.root_atom_idx_list = root_fragment_atom_idx_list
        self.torsion_library_driver_bvh.identify_torsion_mobile_atoms(self.root_atom_idx_list[0])

        root_fragment.SetIntProp('torsion_tree_node_idx', self.current_node_idx)
        root_fragment_visualized.SetIntProp('torsion_tree_node_idx', self.current_node_idx)

        root_fragment_obb_list, root_fragment_obb_info_dict_list = construct_oriented_bounding_box_list(root_fragment)

        self.ligand_torsion_tree.add_node(self.current_node_idx,
                                          parent_node_idx_list=[],
                                          torsion_layer=0,
                                          mol=root_fragment,
                                          fragment_idx=root_fragment_idx,
                                          atom_idx_list=root_fragment_atom_idx_list,
                                          torsion_value_list=[None] * self.num_rotatable_bonds,
                                          obb_list=root_fragment_obb_list,
                                          obb_info_dict_list=root_fragment_obb_info_dict_list,
                                          trimmed_terminal=False)

        if self.create_tree_visualization:
            self.ligand_torsion_tree_visualized.add_node(self.current_node_idx,
                                                         img=mol2image(root_fragment_visualized),
                                                         hac=num_root_atoms)

        self.current_node_idx += 1

        ##############################################################################
        ##############################################################################
        ## Perform deep first search torsion tree construction
        self.__deep_first_search_construction__(0, None)
        ##############################################################################
        ##############################################################################

    def __check_fragment_obb_self_collision__(self, fragment_obb_list, parent_node_idx_list):
        collision_flag = False
        for parent_node_idx in parent_node_idx_list:
            parent_fragment_obb_list = self.ligand_torsion_tree.nodes[parent_node_idx]['obb_list']
            for parent_fragment_obb in parent_fragment_obb_list:
                for fragment_obb in fragment_obb_list:
                    if not fragment_obb.IsOut(parent_fragment_obb):
                        collision_flag = True
                        break

                if collision_flag:
                    break

            if collision_flag:
                break

        return collision_flag

    def __check_fragment_obb_self_collision_temp__(self, fragment_coord_point_list, parent_node_idx_list):
        collision_flag = False
        for parent_node_idx in parent_node_idx_list:
            parent_fragment_obb = self.ligand_torsion_tree.nodes[parent_node_idx]['obb']

            for fragment_coord_point in fragment_coord_point_list:
                if not parent_fragment_obb.IsOut(fragment_coord_point):
                    collision_flag = True
                    break

            if collision_flag:
                break

        return collision_flag

    def __deep_first_search_construction__(self, node_idx, parent_node_idx):
        node = self.ligand_torsion_tree.nodes[node_idx]
        node_parent_node_idx_list = node['parent_node_idx_list']
        node_fragment = node['mol']
        node_fragment_atom_idx_list = node['atom_idx_list']
        node_fragment_idx = node['fragment_idx']
        node_torsion_value_list = node['torsion_value_list']

        if parent_node_idx is None:
            parent_fragment_atom_idx_list = None
        else:
            parent_fragment_atom_idx_list = self.ligand_torsion_tree.nodes[parent_node_idx]['atom_idx_list']

        ##############################################################################
        ##############################################################################
        ## Find children fragments connected to current node fragment
        children_fragment_idx_list = []
        children_atom_idx_nested_list = []
        children_rotatable_bond_info_list = []

        for bond_info in self.rotatable_bond_info_list:
            if bond_info[0] in node_fragment_atom_idx_list:
                if parent_fragment_atom_idx_list is not None and bond_info[1] in parent_fragment_atom_idx_list:
                    continue
                else:
                    parent_atom_idx = bond_info[0]
                    children_atom_idx = bond_info[1]

            elif bond_info[1] in node_fragment_atom_idx_list:
                if parent_fragment_atom_idx_list is not None and bond_info[0] in parent_fragment_atom_idx_list:
                    continue
                else:
                    parent_atom_idx = bond_info[1]
                    children_atom_idx = bond_info[0]

            else:
                continue

            for fragment_idx in range(self.num_fragments):
                fragment_mol = self.fragment_mol_list[fragment_idx]
                fragment_atom_idx_list = [atom.GetIntProp('internal_atom_idx') for atom in fragment_mol.GetAtoms()]

                if children_atom_idx in fragment_atom_idx_list:
                    children_fragment_idx_list.append(fragment_idx)
                    children_atom_idx_nested_list.append(fragment_atom_idx_list)
                    children_rotatable_bond_info_list.append(bond_info)
                    break

        num_offsprings = len(children_fragment_idx_list)
        if num_offsprings == 0:
            node['terminal_rotamer'] = True
        else:
            node['terminal_rotamer'] = False
        ##############################################################################
        ##############################################################################

        ##############################################################################
        ##############################################################################
        ## Construct nodes and edges for all founded children fragments
        for offspring_idx in range(num_offsprings):
            children_fragment_idx = children_fragment_idx_list[offspring_idx]
            children_atom_idx_list = children_atom_idx_nested_list[offspring_idx]
            children_rotatable_bond_info = children_rotatable_bond_info_list[offspring_idx]

            children_fragment_mol = self.fragment_mol_list[children_fragment_idx]
            children_fragment_mol_visualized = self.fragment_mol_list_visualized[children_fragment_idx]
            num_children_fragment_atoms = len(children_atom_idx_list)

            torsion_idx = self.rotatable_bond_info_list.index(children_rotatable_bond_info)
            torsion_angle_value_list = self.torsion_library_driver_bvh.enumerated_torsion_value_nested_list[torsion_idx]
            torsion_atom_idx_list = self.torsion_library_driver_bvh.torsion_atom_idx_nested_list[torsion_idx]

#            for matched_torsion_info_dict in self.torsion_library_driver_bvh.matched_torsion_info_dict_list:
#                matched_rotatable_bond_info = matched_torsion_info_dict['rotatable_bond_info']
#                if matched_rotatable_bond_info == children_rotatable_bond_info:
#                    torsion_angle_value_list = matched_torsion_info_dict['torsion_angle_value']
#                    torsion_atom_idx_list = matched_torsion_info_dict['torsion_atom_idx']
#                    break

            num_possible_torsion_values = len(torsion_angle_value_list)
            generated_children_node_idx_list = []

            for value_idx in range(num_possible_torsion_values):
                torsion_angle_value = torsion_angle_value_list[value_idx]
                specified_torsion_value_list = deepcopy(node_torsion_value_list)
                specified_torsion_value_list[torsion_idx] = torsion_angle_value

                current_children_fragment_mol = deepcopy(children_fragment_mol)
                current_children_fragment_mol.SetIntProp('torsion_tree_node_idx', self.current_node_idx)

                children_obb_list, children_obb_info_dict_list = self.torsion_library_driver_bvh.generate_obb_for_specified_torsion_sets(specified_torsion_value_list,
                                                                                                                                         current_children_fragment_mol)

                current_node_parent_node_idx_list = deepcopy(node_parent_node_idx_list)
                current_node_parent_node_idx_list.append(node_idx)
                current_torsion_layer = len(current_node_parent_node_idx_list)

                if not self.torsion_layer_list[torsion_idx]:
                    self.torsion_layer_list[torsion_idx] = current_torsion_layer
                else:
                    if self.torsion_layer_list[torsion_idx] != current_torsion_layer:
                        raise ValueError('Discrepancies in torsion layer assignments!!')

                if self.__check_fragment_obb_self_collision__(children_obb_list, current_node_parent_node_idx_list):
                    continue

                self.ligand_torsion_tree.add_node(self.current_node_idx,
                                                  parent_node_idx_list=current_node_parent_node_idx_list,
                                                  torsion_layer=current_torsion_layer,
                                                  mol=current_children_fragment_mol,
                                                  fragment_idx=children_fragment_idx,
                                                  atom_idx_list=children_atom_idx_list,
                                                  torsion_value_list=specified_torsion_value_list,
                                                  obb_list=children_obb_list,
                                                  obb_info_dict_list=children_obb_info_dict_list,
                                                  trimmed_terminal=False)

                self.ligand_torsion_tree.add_edge(node_idx,
                                                  self.current_node_idx,
                                                  begin_node_idx=node_idx,
                                                  end_node_idx=self.current_node_idx,
                                                  torsion_atom_idx_list=torsion_atom_idx_list,
                                                  torsion_angle_value=torsion_angle_value)

                if self.create_tree_visualization:
                    current_children_fragment_mol_visualized = deepcopy(children_fragment_mol_visualized)
                    current_children_fragment_mol_visualized.SetIntProp('torsion_tree_node_idx', self.current_node_idx)

                    self.ligand_torsion_tree_visualized.add_node(self.current_node_idx,
                                                                 img=mol2image(current_children_fragment_mol_visualized),
                                                                 hac=num_children_fragment_atoms)

                    self.ligand_torsion_tree_visualized.add_edge(node_idx,
                                                                 self.current_node_idx,
                                                                 begin_node_idx=node_idx,
                                                                 end_node_idx=self.current_node_idx,
                                                                 torsion_atom_idx_list=torsion_atom_idx_list,
                                                                 torsion_angle_value=torsion_angle_value)

                generated_children_node_idx_list.append(self.current_node_idx)
                self.current_node_idx += 1

            if len(generated_children_node_idx_list) == 0:
                node['trimmed_terminal'] = True

                parent_node_atom_idx_str = ''
                for node_fragment_atom_idx in node_fragment_atom_idx_list:
                    parent_node_atom_idx_str += str(node_fragment_atom_idx) + ' '

                parent_node_atom_idx_str.strip()

                children_atom_idx_str = ''
                for children_atom_idx in children_atom_idx_list:
                    children_atom_idx_str += str(children_atom_idx) + ' '

                children_atom_idx_str.strip()

                warnings.warn(f'The parent node atoms: {parent_node_atom_idx_str} collide with all sibling atoms: {children_atom_idx_str}!! Please check torsion library enumerations to make sure this is correct!!')
#                raise ValueError('All enumerated torsion values are invalid under self-collision check!! Please check the rotatable bond definitions and ligand conformations!!')

            ##########################################################################
            ##########################################################################
            ## Continue deep search for current children fragment
            for children_node_idx in generated_children_node_idx_list:
                self.__deep_first_search_construction__(children_node_idx, node_idx)
            ##########################################################################
            ##########################################################################

        ##############################################################################
        ##############################################################################

    def visualize_ligand_torsion_tree(self):
        return create_network_view(self.ligand_torsion_tree_visualized,
                                   color_mapper='',
                                   scale_factor=30,
                                   to_undirected=False,
                                   layout='preset')
