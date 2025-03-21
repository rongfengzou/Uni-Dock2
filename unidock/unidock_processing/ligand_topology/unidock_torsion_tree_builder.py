import os
import warnings
from copy import deepcopy

import networkx as nx

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock.unidock_processing.torsion_library.torsion_rule_matcher import TorsionRuleMatcher
from unidock.unidock_processing.ligand_topology import utils

class UniDockTorsionTreeBuilder(object):
    def __init__(self,
                 mol,
                 root_docked_mol,
                 rotatable_bond_info_list,
                 torsion_library_dict,
                 ligand_file_name_prefix='ligand_unidock',
                 working_dir_name='.'):

        self.mol = mol
        self.root_docked_mol = root_docked_mol
        self.rotatable_bond_info_list = rotatable_bond_info_list
        self.torsion_library_dict = torsion_library_dict
        self.ligand_file_name_prefix = ligand_file_name_prefix
        self.working_dir_name = os.path.abspath(working_dir_name)

        ligand_pdbqt_base_file_name = self.ligand_file_name_prefix + '_torsion_tree.pdbqt'
        self.ligand_pdbqt_file_name = os.path.join(self.working_dir_name, ligand_pdbqt_base_file_name)

        ligand_sdf_base_file_name = self.ligand_file_name_prefix + '_torsion_tree.sdf'
        self.ligand_sdf_file_name = os.path.join(self.working_dir_name, ligand_sdf_base_file_name)

    def assign_root_conformation(self):
        core_atom_mapping_dict = {}
        num_atoms = self.mol.GetNumAtoms()
        for atom_idx in range(num_atoms):
            atom = self.mol.GetAtomWithIdx(atom_idx)
            if atom.HasProp('root_atom_idx'):
                reference_atom_idx = atom.GetIntProp('root_atom_idx')
                core_atom_mapping_dict[reference_atom_idx] = atom_idx

        full_core_atom_mapping_dict = deepcopy(core_atom_mapping_dict)
        for reference_atom_idx in core_atom_mapping_dict:
            query_atom_idx = core_atom_mapping_dict[reference_atom_idx]
            reference_atom = self.root_docked_mol.GetAtomWithIdx(reference_atom_idx)
            query_atom = self.mol.GetAtomWithIdx(query_atom_idx)

            if reference_atom.GetSymbol() == query_atom.GetSymbol():
                reference_neighbor_atom_list = list(reference_atom.GetNeighbors())
                query_neighbor_atom_list = list(query_atom.GetNeighbors())

                if len(reference_neighbor_atom_list) != len(query_neighbor_atom_list):
                    warnings.warn(f'Number of neighbor atoms does not match between query and reference atoms! Please look at this case carefully!')

                reference_neighbor_h_atom_idx_list = []
                query_neighbor_h_atom_idx_list = []

                for reference_neighbor_atom in reference_neighbor_atom_list:
                    if reference_neighbor_atom.GetSymbol() == 'H':
                        reference_neighbor_h_atom_idx_list.append(reference_neighbor_atom.GetIdx())

                for query_neighbor_atom in query_neighbor_atom_list:
                    if query_neighbor_atom.GetSymbol() == 'H':
                        query_neighbor_h_atom_idx_list.append(query_neighbor_atom.GetIdx())

                num_reference_neighbor_h_atoms = len(reference_neighbor_h_atom_idx_list)
                num_query_neighbor_h_atoms = len(query_neighbor_h_atom_idx_list)

                if num_reference_neighbor_h_atoms <= num_query_neighbor_h_atoms:
                    num_neighbor_h_atoms = num_reference_neighbor_h_atoms
                    query_neighbor_h_atom_idx_list = query_neighbor_h_atom_idx_list[:num_reference_neighbor_h_atoms]
                else:
                    num_neighbor_h_atoms = num_query_neighbor_h_atoms
                    reference_neighbor_h_atom_idx_list = reference_neighbor_h_atom_idx_list[:num_query_neighbor_h_atoms]

                for neighbor_h_idx in range(num_neighbor_h_atoms):
                    reference_neighbor_h_atom_idx = reference_neighbor_h_atom_idx_list[neighbor_h_idx]
                    query_neighbor_h_atom_idx = query_neighbor_h_atom_idx_list[neighbor_h_idx]
                    full_core_atom_mapping_dict[reference_neighbor_h_atom_idx] = query_neighbor_h_atom_idx

        self.core_atom_idx_list = utils.get_core_alignment_for_template_docking(self.root_docked_mol, self.mol, full_core_atom_mapping_dict)

        ########################################################################################
        ## Update aligned conformation in atom properties
        ComputeGasteigerCharges(self.mol)
        atom_positions = self.mol.GetConformer().GetPositions()

        for atom_idx in range(num_atoms):
            atom = self.mol.GetAtomWithIdx(atom_idx)
            atom.SetDoubleProp('charge', atom.GetDoubleProp('_GasteigerCharge'))
            atom.SetDoubleProp('x', atom_positions[atom_idx, 0])
            atom.SetDoubleProp('y', atom_positions[atom_idx, 1])
            atom.SetDoubleProp('z', atom_positions[atom_idx, 2])
        ########################################################################################

    def assign_torsion_rules(self):
        torsion_rule_matcher = TorsionRuleMatcher(self.mol,
                                                  self.rotatable_bond_info_list,
                                                  self.torsion_library_dict)

        torsion_rule_matcher.match_torsion_rules()
        self.matched_torsion_info_dict_list = torsion_rule_matcher.matched_torsion_info_dict_list

    def build_molecular_graph(self):
        ########################################################################################
        ## Build mol fragments by rotatable bond info list
        bond_list = list(self.mol.GetBonds())
        rotatable_bond_idx_list = []
        for bond in bond_list:
            bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            if bond_info in self.rotatable_bond_info_list or bond_info_reversed in self.rotatable_bond_info_list:
                rotatable_bond_idx_list.append(bond.GetIdx())

        if len(rotatable_bond_idx_list) != 0:
            splitted_mol = FragmentOnBonds(self.mol, rotatable_bond_idx_list, addDummies=False)
            splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))
        else:
            splitted_mol_list = [self.mol]

        num_fragments = len(splitted_mol_list)
        ########################################################################################

        ########################################################################################
        ## Find fragment as the root node
        num_fragment_atoms_list = [None] * num_fragments
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        root_fragment_idx = None

        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            for atom in fragment.GetAtoms():
                internal_atom_idx = atom.GetIntProp('internal_atom_idx')
                if internal_atom_idx in self.core_atom_idx_list:
                    root_fragment_idx = fragment_idx
                    break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError('Bugs in root finding code for template docking!')
        ########################################################################################

        ########################################################################################
        ## Build torsion tree (Add atom info into nodes)
        torsion_tree = nx.Graph()
        node_idx = 0
        root_fragment = splitted_mol_list[root_fragment_idx]
        num_root_atoms = root_fragment.GetNumAtoms()
        atom_info_list = [None] * num_root_atoms

        for root_atom_idx in range(num_root_atoms):
            root_atom = root_fragment.GetAtomWithIdx(root_atom_idx)
            atom_info_dict = {}
            atom_info_dict['sdf_atom_idx'] = root_atom.GetIntProp('sdf_atom_idx')
            atom_info_dict['atom_name'] = root_atom.GetProp('atom_name')
            atom_info_dict['residue_name'] = root_atom.GetProp('residue_name')
            atom_info_dict['chain_idx'] = root_atom.GetProp('chain_idx')
            atom_info_dict['residue_idx'] = root_atom.GetIntProp('residue_idx')
            atom_info_dict['x'] = root_atom.GetDoubleProp('x')
            atom_info_dict['y'] = root_atom.GetDoubleProp('y')
            atom_info_dict['z'] = root_atom.GetDoubleProp('z')
            atom_info_dict['charge'] = root_atom.GetDoubleProp('charge')
            atom_info_dict['atom_type'] = root_atom.GetProp('atom_type')

            atom_info_list[root_atom_idx] = atom_info_dict

        torsion_tree.add_node(node_idx, atom_info_list=atom_info_list)
        node_idx += 1

        for fragment_idx in range(num_fragments):
            if fragment_idx == root_fragment_idx:
                continue
            else:
                fragment = splitted_mol_list[fragment_idx]
                num_fragment_atoms = fragment.GetNumAtoms()
                atom_info_list = [None] * num_fragment_atoms

                for atom_idx in range(num_fragment_atoms):
                    atom = fragment.GetAtomWithIdx(atom_idx)
                    atom_info_dict = {}
                    atom_info_dict['sdf_atom_idx'] = atom.GetIntProp('sdf_atom_idx')
                    atom_info_dict['atom_name'] = atom.GetProp('atom_name')
                    atom_info_dict['residue_name'] = atom.GetProp('residue_name')
                    atom_info_dict['chain_idx'] = atom.GetProp('chain_idx')
                    atom_info_dict['residue_idx'] = atom.GetIntProp('residue_idx')
                    atom_info_dict['x'] = atom.GetDoubleProp('x')
                    atom_info_dict['y'] = atom.GetDoubleProp('y')
                    atom_info_dict['z'] = atom.GetDoubleProp('z')
                    atom_info_dict['charge'] = atom.GetDoubleProp('charge')
                    atom_info_dict['atom_type'] = atom.GetProp('atom_type')

                    atom_info_list[atom_idx] = atom_info_dict

                torsion_tree.add_node(node_idx, atom_info_list=atom_info_list)
                node_idx += 1

        ########################################################################################

        ########################################################################################
        ## Build torsion tree (Add edge info into nodes)
        num_rotatable_bonds = len(self.rotatable_bond_info_list)
        for edge_idx in range(num_rotatable_bonds):
            rotatable_bond_info = self.rotatable_bond_info_list[edge_idx]
            begin_atom_idx = rotatable_bond_info[0]
            end_atom_idx = rotatable_bond_info[1]

            begin_atom = self.mol.GetAtomWithIdx(begin_atom_idx)
            begin_sdf_atom_idx = begin_atom.GetIntProp('sdf_atom_idx')
            begin_atom_name = begin_atom.GetProp('atom_name')

            end_atom = self.mol.GetAtomWithIdx(end_atom_idx)
            end_sdf_atom_idx = end_atom.GetIntProp('sdf_atom_idx')
            end_atom_name = end_atom.GetProp('atom_name')

            begin_node_idx = None
            end_node_idx = None
            for node_idx in range(num_fragments):
                atom_info_list = torsion_tree.nodes[node_idx]['atom_info_list']
                for atom_info_dict in atom_info_list:
                    if atom_info_dict['atom_name'] == begin_atom_name:
                        begin_node_idx = node_idx
                        break
                    elif atom_info_dict['atom_name'] == end_atom_name:
                        end_node_idx = node_idx
                        break

                if begin_node_idx is not None and end_node_idx is not None:
                    break

            if begin_node_idx is None or end_node_idx is None:
                raise ValueError('Bugs in edge assignment code!!')

            torsion_tree.add_edge(begin_node_idx,
                                  end_node_idx,
                                  begin_node_idx=begin_node_idx,
                                  end_node_idx=end_node_idx,
                                  begin_sdf_atom_idx=begin_sdf_atom_idx,
                                  end_sdf_atom_idx=end_sdf_atom_idx,
                                  begin_atom_name=begin_atom_name,
                                  end_atom_name=end_atom_name)
        ##############################################################################

        self.torsion_tree = torsion_tree

    def __deep_first_search__(self, node_idx):
        if node_idx == 0:
            self.pdbqt_atom_line_list.append('ROOT\n')
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                atom_name = atom_info_dict['atom_name']
                self.atom_idx_info_mapping_dict[atom_name] = self.pdbqt_atom_idx
                atom_info_tuple = ('ATOM',
                                   self.pdbqt_atom_idx,
                                   atom_info_dict['atom_name'],
                                   atom_info_dict['residue_name'],
                                   atom_info_dict['chain_idx'],
                                   atom_info_dict['residue_idx'],
                                   atom_info_dict['x'],
                                   atom_info_dict['y'],
                                   atom_info_dict['z'],
                                   1.0,
                                   0.0,
                                   atom_info_dict['charge'],
                                   atom_info_dict['atom_type'])

                self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))
                self.pdbqt_atom_idx += 1

            self.pdbqt_atom_line_list.append('ENDROOT\n')

        else:
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                atom_name = atom_info_dict['atom_name']
                if atom_name not in self.atom_idx_info_mapping_dict:
                    self.atom_idx_info_mapping_dict[atom_name] = self.pdbqt_atom_idx

                atom_info_tuple = ('ATOM',
                                   self.pdbqt_atom_idx,
                                   atom_info_dict['atom_name'],
                                   atom_info_dict['residue_name'],
                                   atom_info_dict['chain_idx'],
                                   atom_info_dict['residue_idx'],
                                   atom_info_dict['x'],
                                   atom_info_dict['y'],
                                   atom_info_dict['z'],
                                   1.0,
                                   0.0,
                                   atom_info_dict['charge'],
                                   atom_info_dict['atom_type'])

                self.pdbqt_atom_line_list.append(self.pdbqt_atom_line_format.format(*atom_info_tuple))
                self.pdbqt_atom_idx += 1

        self.visited_node_idx_set.add(node_idx)

        neighbor_node_idx_list = list(self.torsion_tree.neighbors(node_idx))
        for neighbor_node_idx in neighbor_node_idx_list:
            if neighbor_node_idx not in self.visited_node_idx_set:
                temp_pdbqt_atom_idx = self.pdbqt_atom_idx
                atom_info_list = self.torsion_tree.nodes[neighbor_node_idx]['atom_info_list']
                for atom_info_dict in atom_info_list:
                    atom_name = atom_info_dict['atom_name']
                    if atom_name not in self.atom_idx_info_mapping_dict:
                        self.atom_idx_info_mapping_dict[atom_name] = temp_pdbqt_atom_idx
                        temp_pdbqt_atom_idx += 1

                edge_info = self.torsion_tree.edges[(node_idx, neighbor_node_idx)]
                begin_node_idx = edge_info['begin_node_idx']
                end_node_idx = edge_info['end_node_idx']
                begin_atom_name = edge_info['begin_atom_name']
                end_atom_name = edge_info['end_atom_name']

                if begin_node_idx == node_idx:
                    parent_atom_name = begin_atom_name
                    offspring_atom_name = end_atom_name
                else:
                    parent_atom_name = end_atom_name
                    offspring_atom_name = begin_atom_name

                parent_atom_idx = self.atom_idx_info_mapping_dict[parent_atom_name]
                offspring_atom_idx = self.atom_idx_info_mapping_dict[offspring_atom_name]

                self.branch_info_list.append((parent_atom_name, str(parent_atom_idx), offspring_atom_name, str(offspring_atom_idx)))
                self.pdbqt_atom_line_list.append(self.pdbqt_branch_line_format.format('BRANCH', parent_atom_idx, offspring_atom_idx))
                self.__deep_first_search__(neighbor_node_idx)
                self.pdbqt_atom_line_list.append(self.pdbqt_end_branch_line_format.format('ENDBRANCH', parent_atom_idx, offspring_atom_idx))

    def write_pdbqt_file(self):
        self.pdbqt_remark_line_list = []
        self.pdbqt_atom_line_list = []

        self.pdbqt_remark_torsion_line_format = '{:6s}   {:^2d}  {:1s}    {:7s} {:6s} {:^7s}  {:3s}  {:^7s}\n'
        self.pdbqt_atom_line_format = '{:4s}  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n'
        self.pdbqt_branch_line_format = '{:6s} {:3d} {:3d}\n'
        self.pdbqt_end_branch_line_format = '{:9s} {:3d} {:3d}\n'
        self.torsion_dof_line_format = '{:7s} {:d}'

        ## Prepare pdbqt atom lines
        ####################################################################################################
        self.atom_idx_info_mapping_dict = {}
        self.branch_info_list = []
        self.visited_node_idx_set = set()
        self.pdbqt_atom_idx = 1

        self.__deep_first_search__(0)
        self.num_torsions = len(self.branch_info_list)
        self.pdbqt_atom_line_list.append(self.torsion_dof_line_format.format('TORSDOF', self.num_torsions))
        ####################################################################################################

        ## Prepare pdbqt remark lines
        ####################################################################################################
        self.pdbqt_remark_line_list.append('REMARK  '  + str(self.num_torsions) + ' active torsions:\n')
        self.pdbqt_remark_line_list.append("REMARK  status: ('A' for Active; 'I' for Inactive)\n")
        for torsion_idx in range(self.num_torsions):
            branch_info_tuple = self.branch_info_list[torsion_idx]
            remark_torsion_info_tuple = ('REMARK',
                                         torsion_idx+1,
                                         'A',
                                         'between',
                                         'atoms:',
                                         branch_info_tuple[0] + '_' + branch_info_tuple[1],
                                         'and',
                                         branch_info_tuple[2] + '_' + branch_info_tuple[3])

            self.pdbqt_remark_line_list.append(self.pdbqt_remark_torsion_line_format.format(*remark_torsion_info_tuple))
        ####################################################################################################

        self.pdbqt_line_list = self.pdbqt_remark_line_list + self.pdbqt_atom_line_list

        with open(self.ligand_pdbqt_file_name, 'w') as ligand_pdbqt_file:
            for pdbqt_line in self.pdbqt_line_list:
                ligand_pdbqt_file.write(pdbqt_line)

    def write_torsion_tree_sdf_file(self):
        fragment_info_string = ''
        torsion_info_string = ''
        atom_info_string = ''
        torsion_sampling_range_info_string = ''

        num_nodes = self.torsion_tree.number_of_nodes()
        num_edges = self.torsion_tree.number_of_edges()

        ####################################################################################################
        ## Fragment info
        for node_idx in range(num_nodes):
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                fragment_info_string += str(atom_info_dict['sdf_atom_idx'])
                fragment_info_string += ' '

            fragment_info_string = fragment_info_string[:-1]
            fragment_info_string += '\n'
        ####################################################################################################

        ####################################################################################################
        ## Torsion info
        edge_key_list = list(self.torsion_tree.edges.keys())
        for edge_idx in range(num_edges):
            edge_key = edge_key_list[edge_idx]
            edge_info_dict = self.torsion_tree.edges[edge_key]
            begin_sdf_atom_idx = str(edge_info_dict['begin_sdf_atom_idx'])
            end_sdf_atom_idx = str(edge_info_dict['end_sdf_atom_idx'])
            begin_node_idx = str(edge_info_dict['begin_node_idx'])
            end_node_idx = str(edge_info_dict['end_node_idx'])

            torsion_info_string += f'{begin_sdf_atom_idx} {end_sdf_atom_idx} {begin_node_idx} {end_node_idx}'
            torsion_info_string += '\n'
        ####################################################################################################

        ####################################################################################################
        ## Atom info
        atom_info_line_format = '{:<3d}{:<10f}{:<2s}\n'
        for atom in self.mol.GetAtoms():
            sdf_atom_idx = atom.GetIntProp('sdf_atom_idx')
            charge = atom.GetDoubleProp('charge')
            atom_type = atom.GetProp('atom_type')

            atom_info_string += atom_info_line_format.format(sdf_atom_idx, charge, atom_type)
        ####################################################################################################

        ####################################################################################################
        ## Torsion range info
        for torsion_info_dict in self.matched_torsion_info_dict_list:
            torsion_atom_idx_tuple = torsion_info_dict['torsion_atom_idx']
            torsion_smarts = torsion_info_dict['torsion_smarts']
            torsion_angle_range_list = torsion_info_dict['torsion_angle_range']
            torsion_histogram_weight_list = torsion_info_dict['torsion_histogram_weight'] 

            torsion_atom_idx_str = ' '.join([str(torsion_atom_idx + 1) for torsion_atom_idx in torsion_atom_idx_tuple])
            torsion_sampling_range_info_string += f'TORSION {torsion_atom_idx_str}'
            torsion_sampling_range_info_string += '\n'

            torsion_sampling_range_info_string += f'SMARTS {torsion_smarts}'
            torsion_sampling_range_info_string += '\n'

            for torsion_angle_range in torsion_angle_range_list:
                torsion_angle_range_str = ' '.join([str(torsion_angle_range[0]), str(torsion_angle_range[1])])
                torsion_sampling_range_info_string += f'RANGE {torsion_angle_range_str}'
                torsion_sampling_range_info_string += '\n'

            torsion_historgram_weight_str = ' '.join([str(torsion_histogram_weight) for torsion_histogram_weight in torsion_histogram_weight_list])
            torsion_sampling_range_info_string += f'HISTOGRAM {torsion_historgram_weight_str}'
            torsion_sampling_range_info_string += '\n'
        ####################################################################################################

        self.mol.SetProp('fragInfo', fragment_info_string)
        self.mol.SetProp('torsionInfo', torsion_info_string)
        self.mol.SetProp('atomInfo', atom_info_string)
        self.mol.SetProp('torsionSamplingRangeInfo', torsion_sampling_range_info_string)

        writer = Chem.SDWriter(self.ligand_sdf_file_name)
        writer.write(self.mol)
        writer.flush()
        writer.close()
