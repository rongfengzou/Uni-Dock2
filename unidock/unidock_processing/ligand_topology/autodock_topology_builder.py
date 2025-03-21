import os
import re

import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock.unidock_processing.ligand_topology.atom_type import AtomType
from unidock.unidock_processing.ligand_topology.generic_rotatable_bond import GenericRotatableBond
from unidock.unidock_processing.ligand_topology import utils

class AutoDockTopologyBuilder(object):
    def __init__(self,
                 ligand_sdf_file_name,
                 covalent_ligand=False,
                 template_docking=False,
                 reference_sdf_file_name=None,
                 core_atom_mapping_dict=None,
                 working_dir_name='.'):

        self.ligand_sdf_file_name = ligand_sdf_file_name
        self.covalent_ligand = covalent_ligand
        self.template_docking = template_docking

        if reference_sdf_file_name is not None:
            self.reference_sdf_file_name = os.path.abspath(reference_sdf_file_name)
        else:
            self.reference_sdf_file_name = None

        self.core_atom_mapping_dict = core_atom_mapping_dict

        if self.template_docking:
            if self.reference_sdf_file_name is None:
                raise ValueError('template docking mode specified without reference SDF file!!')
            else:
                self.reference_mol = Chem.SDMolSupplier(self.reference_sdf_file_name, removeHs=False)[0]
        else:
            self.reference_mol = None

        self.working_dir_name = os.path.abspath(working_dir_name)

        ligand_sdf_base_file_name = os.path.basename(self.ligand_sdf_file_name)
        ligand_file_name_prefix = ligand_sdf_base_file_name.split('.')[0]

        ligand_pdbqt_base_file_name = ligand_file_name_prefix + '_torsion_tree.pdbqt'
        self.ligand_pdbqt_file_name = os.path.join(self.working_dir_name, ligand_pdbqt_base_file_name)

        ligand_torsion_tree_sdf_base_file_name = ligand_file_name_prefix + '_torsion_tree.sdf'
        self.ligand_torsion_tree_sdf_file_name = os.path.join(self.working_dir_name, ligand_torsion_tree_sdf_base_file_name)

        if self.template_docking:
            ligand_core_bpf_base_file_name = ligand_file_name_prefix + '_torsion_tree.bpf'
            self.ligand_core_bpf_file_name = os.path.join(self.working_dir_name, ligand_core_bpf_base_file_name)

        self.atom_typer = AtomType()
        self.rotatable_bond_finder = GenericRotatableBond()

    def build_molecular_graph(self):
        mol = Chem.SDMolSupplier(self.ligand_sdf_file_name, removeHs=False)[0]

        if self.covalent_ligand:
            mol, covalent_anchor_atom_info, covalent_atom_info_list = utils.prepare_covalent_ligand_mol(mol)
        else:
            covalent_anchor_atom_info = None
            covalent_atom_info_list = None

        if self.template_docking:
            if self.core_atom_mapping_dict is None:
                self.core_atom_mapping_dict = utils.get_template_docking_atom_mapping(self.reference_mol, mol)
            else:
                if not utils.check_manual_atom_mapping_connection(self.reference_mol, mol, self.core_atom_mapping_dict):
                    raise ValueError('Specified core atom mapping makes unconnected fragments!!')

            core_atom_idx_list = utils.get_core_alignment_for_template_docking(self.reference_mol, mol, self.core_atom_mapping_dict)
            for core_atom_idx in core_atom_idx_list:
                core_atom = mol.GetAtomWithIdx(core_atom_idx)
                for neighbor_atom in core_atom.GetNeighbors():
                    if neighbor_atom.GetIdx() not in core_atom_idx_list:
                        core_atom_idx_list.remove(core_atom_idx)
                        break

        else:
            core_atom_idx_list = None

        self.atom_typer.assign_atom_types(mol)
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(mol)

        ## Freeze bonds in covalent part to be unrotatable for covalent ligand case
        ##############################################################################
#        if self.covalent_ligand:
#            covalent_atom_idx_list = []
#            num_atoms = mol.GetNumAtoms()
#            for atom_idx in range(num_atoms):
#                atom = mol.GetAtomWithIdx(atom_idx)
#                if atom.GetProp('residue_name') != 'MOL':
#                    covalent_atom_idx_list.append(atom_idx)
#
#            filtered_rotatable_bond_info_list = []
#            for rotatable_bond_info in rotatable_bond_info_list:
#                if rotatable_bond_info[0] in covalent_atom_idx_list and rotatable_bond_info[1] in covalent_atom_idx_list:
#                    continue
#                else:
#                    filtered_rotatable_bond_info_list.append(rotatable_bond_info)
#
#            rotatable_bond_info_list = filtered_rotatable_bond_info_list
        ##############################################################################

        ## Freeze bonds in core part to be unrotatable for template docking case
        ###############################################################################
        if self.template_docking:
            filtered_rotatable_bond_info_list = []
            for rotatable_bond_info in rotatable_bond_info_list:
                rotatable_begin_atom_idx = rotatable_bond_info[0]
                rotatable_end_atom_idx = rotatable_bond_info[1]
                if rotatable_begin_atom_idx in core_atom_idx_list and rotatable_end_atom_idx in core_atom_idx_list:
                    continue
                else:
                    filtered_rotatable_bond_info_list.append(rotatable_bond_info)

            rotatable_bond_info_list = filtered_rotatable_bond_info_list
        ###############################################################################

        bond_list = list(mol.GetBonds())
        rotatable_bond_idx_list = []
        for bond in bond_list:
            bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            if bond_info in rotatable_bond_info_list or bond_info_reversed in rotatable_bond_info_list:
                rotatable_bond_idx_list.append(bond.GetIdx())

        if len(rotatable_bond_idx_list) != 0:
            splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
            splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))
        else:
            splitted_mol_list = [mol]

        num_fragments = len(splitted_mol_list)

        ## Find fragment as the root node
        ##############################################################################
        num_fragment_atoms_list = [None] * num_fragments
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        root_fragment_idx = None
        if self.covalent_ligand:
            for fragment_idx in range(num_fragments):
                fragment = splitted_mol_list[fragment_idx]
                for atom in fragment.GetAtoms():
                    atom_info = (atom.GetProp('chain_idx'), atom.GetProp('residue_name'), atom.GetIntProp('residue_idx'), atom.GetProp('atom_name'))
                    if atom_info == covalent_anchor_atom_info:
                        root_fragment_idx = fragment_idx
                        break

                if root_fragment_idx is not None:
                    break

            if root_fragment_idx is None:
                raise ValueError('Bugs in root finding code for covalent docking!')

        elif self.template_docking:
            for fragment_idx in range(num_fragments):
                fragment = splitted_mol_list[fragment_idx]
                for atom in fragment.GetAtoms():
                    internal_atom_idx = int(re.split(r'(\d+)', atom.GetProp('atom_name'))[1]) - 1
                    if internal_atom_idx in core_atom_idx_list:
                        root_fragment_idx = fragment_idx
                        break

                if root_fragment_idx is not None:
                    break

            if root_fragment_idx is None:
                raise ValueError('Bugs in root finding code for template docking!')

        else:
            root_fragment_idx = np.argmax(num_fragment_atoms_list)
        ##############################################################################

        ## Build torsion tree
        ### Add atom info into nodes
        ##############################################################################
        torsion_tree = nx.Graph()
        node_idx = 0
        root_fragment = splitted_mol_list[root_fragment_idx]
        num_root_atoms = root_fragment.GetNumAtoms()
        atom_info_list = [None] * num_root_atoms

        self.root_atom_idx_list = []

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
            atom_info_dict['atom_type'] = root_atom.GetProp('vina_atom_type')

            atom_info_list[root_atom_idx] = atom_info_dict

            self.root_atom_idx_list.append(root_atom.GetIntProp('internal_atom_idx'))

        ##############################################################################
        ##############################################################################
        ## FIXME: Workaround for mol produced by unordered reaction smarts 
        if self.covalent_ligand:
            reordered_atom_info_list = []
            num_covalent_residue_atoms = len(covalent_atom_info_list)

            for covalent_residue_atom_idx in range(num_covalent_residue_atoms):
                covalent_atom_info_tuple =  covalent_atom_info_list[covalent_residue_atom_idx]

                for atom_info_dict in atom_info_list:
                    atom_info_tuple = (atom_info_dict['chain_idx'],
                                       atom_info_dict['residue_name'],
                                       atom_info_dict['residue_idx'],
                                       atom_info_dict['atom_name'])

                    if atom_info_tuple == covalent_atom_info_tuple:
                        reordered_atom_info_list.append(atom_info_dict)

            atom_info_list = reordered_atom_info_list
            ##############################################################################
            ##############################################################################

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
                    atom_info_dict['atom_type'] = atom.GetProp('vina_atom_type')

                    atom_info_list[atom_idx] = atom_info_dict

                torsion_tree.add_node(node_idx, atom_info_list=atom_info_list)
                node_idx += 1

        ##############################################################################

        ### Add edge info
        ##############################################################################
        num_rotatable_bonds = len(rotatable_bond_info_list)
        for edge_idx in range(num_rotatable_bonds):
            rotatable_bond_info = rotatable_bond_info_list[edge_idx]
            begin_atom_idx = rotatable_bond_info[0]
            end_atom_idx = rotatable_bond_info[1]

            begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
            begin_sdf_atom_idx = begin_atom.GetIntProp('sdf_atom_idx')
            begin_atom_name = begin_atom.GetProp('atom_name')

            end_atom = mol.GetAtomWithIdx(end_atom_idx)
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
        self.mol = mol

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
        self.pdbqt_atom_line_format = '{:4s}  {:5d} {:^4s} {:4s}{:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n'
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
        self.root_atom_idx_str = ','.join([str(root_atom_idx) for root_atom_idx in self.root_atom_idx_list])
        ####################################################################################################

        ## Prepare pdbqt remark lines
        ####################################################################################################
        self.pdbqt_remark_line_list.append('REMARK  '  + str(self.num_torsions) + ' active torsions:\n')
        self.pdbqt_remark_line_list.append("REMARK  status: ('A' for Active; 'I' for Inactive)\n")

        if self.covalent_ligand or self.template_docking:
            self.pdbqt_remark_line_list.append('REMARK  TEMPLATE ROOT ATOMS: ' + self.root_atom_idx_str + '\n')

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

    def write_constraint_bpf_file(self):
        self.core_bpf_remark_line_list = []
        self.core_bpf_atom_line_list = []
        self.core_bpf_atom_line_format = '{:8.3f}\t{:8.3f}\t{:8.3f}\t{:6.2f}\t{:6.2f}\t{:3s}\t{:<2s}\n'

        self.core_bpf_remark_line_list.append('x y z Vset r type atom\n')

        root_atom_info_list = self.torsion_tree.nodes[0]['atom_info_list']
        for atom_info_dict in root_atom_info_list:
            atom_info_tuple = (atom_info_dict['x'],
                               atom_info_dict['y'],
                               atom_info_dict['z'],
                               -1.2,
                               0.6,
                               'map',
                               atom_info_dict['atom_type'])

            self.core_bpf_atom_line_list.append(self.core_bpf_atom_line_format.format(*atom_info_tuple))

        self.core_bpf_line_list = self.core_bpf_remark_line_list + self.core_bpf_atom_line_list

        with open(self.ligand_core_bpf_file_name, 'w') as ligand_core_bpf_file:
            for core_bpf_line in self.core_bpf_line_list:
                ligand_core_bpf_file.write(core_bpf_line)

    def write_torsion_tree_sdf_file(self):
        fragment_info_string = ''
        torsion_info_string = ''
        atom_info_string = ''

        num_nodes = self.torsion_tree.number_of_nodes()
        num_edges = self.torsion_tree.number_of_edges()

        for node_idx in range(num_nodes):
            atom_info_list = self.torsion_tree.nodes[node_idx]['atom_info_list']
            for atom_info_dict in atom_info_list:
                fragment_info_string += str(atom_info_dict['sdf_atom_idx'])
                fragment_info_string += ' '

            fragment_info_string = fragment_info_string[:-1]
            fragment_info_string += '\n'

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

        atom_info_line_format = '{:<3d}{:<10f}{:<2s}\n'
        for atom in self.mol.GetAtoms():
            sdf_atom_idx = atom.GetIntProp('sdf_atom_idx')
            charge = atom.GetDoubleProp('charge')
            atom_type = atom.GetProp('vina_atom_type')

            atom_info_string += atom_info_line_format.format(sdf_atom_idx, charge, atom_type)

        self.mol.SetProp('fragInfo', fragment_info_string)
        self.mol.SetProp('torsionInfo', torsion_info_string)
        self.mol.SetProp('atomInfo', atom_info_string)

        writer = Chem.SDWriter(self.ligand_torsion_tree_sdf_file_name)
        writer.write(self.mol)
        writer.flush()
        writer.close()
