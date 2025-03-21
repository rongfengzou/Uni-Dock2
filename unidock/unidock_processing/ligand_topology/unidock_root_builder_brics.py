import os
from copy import deepcopy
from tqdm.notebook import tqdm

import numpy as np
import networkx as nx

import multiprocess as mp
from multiprocess.pool import Pool

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds, rdMolTransforms
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Geometry import Point3D
from rdkit.Chem.BRICS import FindBRICSBonds

from unidock.unidock_processing.ligand_topology.atom_type import AtomType
from unidock.unidock_processing.ligand_topology.rotatable_bond import RotatableBond
from unidock.unidock_processing.ligand_topology import utils

from unidock.unidock_processing.torsion_library.torsion_library_driver_old import TorsionLibraryDriver

def batch_write_root_fragment_process(batch_root_sdf_file_name_list, batch_root_fragment_conf_mol_list):
    num_batch_conformations = len(batch_root_sdf_file_name_list)
    for conf_idx in range(num_batch_conformations):
        root_sdf_file_name = batch_root_sdf_file_name_list[conf_idx]
        root_fragment_conf_mol = batch_root_fragment_conf_mol_list[conf_idx]

        writer = Chem.SDWriter(root_sdf_file_name)
        writer.write(root_fragment_conf_mol)
        writer.flush()
        writer.close()

class UniDockRootBuilder(object):
    def __init__(self,
                 ligand_sdf_file_name,
                 torsion_library_dict,
                 target_center=(0.0, 0.0, 0.0),
                 max_num_root_rotatable_bonds=4,
                 max_num_root_torsion_enumerations=8,
                 n_cpu=16,
                 working_dir_name='.'):

        self.ligand_sdf_file_name = ligand_sdf_file_name
        self.torsion_library_dict = torsion_library_dict
        self.target_center = np.array(target_center)
        self.max_num_root_rotatable_bonds = max_num_root_rotatable_bonds
        self.max_num_root_torsion_enumerations = max_num_root_torsion_enumerations
        self.n_cpu = n_cpu
        self.working_dir_name = os.path.abspath(working_dir_name)

        self.atom_typer = AtomType()
        self.rotatable_bond_finder = RotatableBond()

    def __deep_first_search__(self, node_idx):
        offspring_node_idx_list = list(self.torsion_tree.neighbors(node_idx))
        num_offspring_nodes = len(offspring_node_idx_list)
        num_offspring_node_atoms_list = [None] * num_offspring_nodes

        for offspring_idx in range(num_offspring_nodes):
            offspring_node_idx = offspring_node_idx_list[offspring_idx]
            offspring_atom_info_list = self.torsion_tree.nodes[offspring_node_idx]['atom_info_list']
            num_offspring_node_atoms_list[offspring_idx] = len(offspring_atom_info_list)

        offspring_node_idx_array = np.array(offspring_node_idx_list, dtype=np.int32)
        num_offspring_node_atoms_array = np.array(num_offspring_node_atoms_list, dtype=np.int32)

        sorted_offspring_idx_array = np.argsort(num_offspring_node_atoms_array)[::-1]
        sorted_offspring_node_idx_array = offspring_node_idx_array[sorted_offspring_idx_array]
        sorted_num_offspring_node_atoms_array = num_offspring_node_atoms_array[sorted_offspring_idx_array]

        for offspring_idx in range(num_offspring_nodes):
            if len(self.root_rotatable_bond_info_list) == self.max_num_root_rotatable_bonds:
                break

            offspring_node_idx = sorted_offspring_node_idx_array[offspring_idx]
            if offspring_node_idx not in self.visited_node_idx_set:
                self.visited_node_idx_set.add(offspring_node_idx)
                edge_info = self.torsion_tree.edges[(node_idx, offspring_node_idx)]
                rotatable_bond_info = (edge_info['begin_atom_idx'], edge_info['end_atom_idx'])

                if rotatable_bond_info not in self.root_rotatable_bond_info_list:
                    self.root_rotatable_bond_info_list.append(rotatable_bond_info)
                else:
                    raise ValueError('Current rotatable bond visited!! Bugs in dfs code!!')

                if len(self.root_rotatable_bond_info_list) == self.max_num_root_rotatable_bonds:
                    break

                self.__deep_first_search__(offspring_node_idx)

    def build_root_fragment(self):
        mol = Chem.SDMolSupplier(self.ligand_sdf_file_name, removeHs=False)[0]

        self.atom_typer.assign_atom_types(mol)
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)

        #############################################################################################
        ## Identify rotatable bonds using internal rules and BRICS decomposition method
        internal_rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(mol)
        brics_rotatable_bond_info_raw_list = list(FindBRICSBonds(mol))
        brics_rotatable_bond_info_list = []
        for brics_rotatable_bond_info_raw in brics_rotatable_bond_info_raw_list:
            brics_rotatable_bond_info_list.append(brics_rotatable_bond_info_raw[0])

        self.rotatable_bond_info_list = []
        for rotatable_bond_info in internal_rotatable_bond_info_list:
            rotatable_bond_info_reversed = (rotatable_bond_info[1], rotatable_bond_info[0])
            if rotatable_bond_info in brics_rotatable_bond_info_list or rotatable_bond_info_reversed in brics_rotatable_bond_info_list:
                self.rotatable_bond_info_list.append(rotatable_bond_info)

        #############################################################################################

        #############################################################################################
        ## Fragmentize molecule by rotatable bonds
        bond_list = list(mol.GetBonds())
        rotatable_bond_idx_list = []
        for bond in bond_list:
            bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            if bond_info in self.rotatable_bond_info_list or bond_info_reversed in self.rotatable_bond_info_list:
                rotatable_bond_idx_list.append(bond.GetIdx())

        if len(rotatable_bond_idx_list) > self.max_num_root_rotatable_bonds:
            self.rigid_ligand = False
            splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
            splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))
        else:
            self.rigid_ligand = True
            splitted_mol_list = [mol]

        num_fragments = len(splitted_mol_list)
        #############################################################################################

        #############################################################################################
        ## Find root fragment
        num_fragment_atoms_list = [None] * num_fragments
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        root_fragment_idx = np.argmax(num_fragment_atoms_list)
        #############################################################################################

        #############################################################################################
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
        #############################################################################################

        #############################################################################################
        ## Build torsion tree (Add edge info into nodes)
        if not self.rigid_ligand:
            num_rotatable_bonds = len(self.rotatable_bond_info_list)
            for edge_idx in range(num_rotatable_bonds):
                rotatable_bond_info = self.rotatable_bond_info_list[edge_idx]
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
                                      begin_atom_idx=begin_atom_idx,
                                      end_atom_idx=end_atom_idx,
                                      begin_sdf_atom_idx=begin_sdf_atom_idx,
                                      end_sdf_atom_idx=end_sdf_atom_idx,
                                      begin_atom_name=begin_atom_name,
                                      end_atom_name=end_atom_name)

        self.torsion_tree = torsion_tree
        #############################################################################################

        #############################################################################################
        ## Perform DFS to get real rotamer rotatable bonds
        if self.rigid_ligand:
            self.rotamer_rotatable_bond_info_list = []
            self.root_rotatable_bond_info_list = self.rotatable_bond_info_list
        else:
            self.visited_node_idx_set = set()
            self.root_rotatable_bond_info_list = []
            num_rotatable_bonds = len(self.rotatable_bond_info_list)
            self.max_num_root_rotatable_bonds = min(num_rotatable_bonds, self.max_num_root_rotatable_bonds)

            self.visited_node_idx_set.add(0)
            self.__deep_first_search__(0)

            self.rotamer_rotatable_bond_info_list = []
            for rotatable_bond_info in self.rotatable_bond_info_list:
                if rotatable_bond_info not in self.root_rotatable_bond_info_list:
                    self.rotamer_rotatable_bond_info_list.append(rotatable_bond_info)
        #############################################################################################

        #############################################################################################
        ## Redo fragmentization and get the expanded root fragment
        if self.rigid_ligand:
            expanded_root_fragment = mol
        else:
            rotamer_rotatable_bond_idx_list = []
            bond_list = list(mol.GetBonds())
            for bond in bond_list:
                bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                if bond_info in self.rotamer_rotatable_bond_info_list or bond_info_reversed in self.rotamer_rotatable_bond_info_list:
                    rotamer_rotatable_bond_idx_list.append(bond.GetIdx())

            if len(rotamer_rotatable_bond_idx_list) > 0:
                rotamer_splitted_mol = FragmentOnBonds(mol, rotamer_rotatable_bond_idx_list, addDummies=False)
                rotamer_splitted_mol_list = list(GetMolFrags(rotamer_splitted_mol, asMols=True, sanitizeFrags=False))
            else:
                raise ValueError('Bugs in rotamer group assignment codes!!')

            num_fragments = len(rotamer_splitted_mol_list)
            num_fragment_atoms_list = [None] * num_fragments
            for fragment_idx in range(num_fragments):
                fragment = rotamer_splitted_mol_list[fragment_idx]
                num_atoms = fragment.GetNumAtoms()
                num_fragment_atoms_list[fragment_idx] = num_atoms

            expanded_root_fragment_idx = np.argmax(num_fragment_atoms_list)
            expanded_root_fragment = rotamer_splitted_mol_list[expanded_root_fragment_idx]
        #############################################################################################

        #############################################################################################
        ## Assign heavy atom idx for root fragments for later whole mol atom mapping
        heavy_atom_idx = 0
        for atom in expanded_root_fragment.GetAtoms():
            if atom.GetSymbol() != 'H':
                atom.SetIntProp('root_heavy_atom_idx', heavy_atom_idx)
                heavy_atom_idx += 1
        #############################################################################################

        #############################################################################################
        ## Add capped fragments on root fragment
        if self.rigid_ligand:
            expanded_root_fragment_capped = mol
        else:
            num_root_atoms_temp = expanded_root_fragment.GetNumAtoms()
            expanded_root_fragment_H = Chem.AddHs(expanded_root_fragment, addCoords=True)
            num_root_atoms_temp_H = expanded_root_fragment_H.GetNumAtoms()

            conformer = expanded_root_fragment_H.GetConformer()
            for atom_idx in range(num_root_atoms_temp, num_root_atoms_temp_H):
                capped_H_atom = expanded_root_fragment_H.GetAtomWithIdx(atom_idx)
                capped_H_atom.SetAtomicNum(6)

                Chem.GetSymmSSSR(expanded_root_fragment_H)
                expanded_root_fragment_H.UpdatePropertyCache(strict=False)

                capped_H_neighbor_atom = list(capped_H_atom.GetNeighbors())[0]
                capped_H_neighbor_atom_idx = capped_H_neighbor_atom.GetIdx()
                capped_H_neighbor_element = capped_H_neighbor_atom.GetSymbol()

                if capped_H_neighbor_element == 'C':
                    rdMolTransforms.SetBondLength(conformer, capped_H_neighbor_atom_idx, atom_idx, 1.54)
                elif capped_H_neighbor_element == 'O':
                    rdMolTransforms.SetBondLength(conformer, capped_H_neighbor_atom_idx, atom_idx, 1.43)
                elif capped_H_neighbor_element == 'N':
                    rdMolTransforms.SetBondLength(conformer, capped_H_neighbor_atom_idx, atom_idx, 1.475)
                elif capped_H_neighbor_element == 'S':
                    rdMolTransforms.SetBondLength(conformer, capped_H_neighbor_atom_idx, atom_idx, 1.60)
                elif capped_H_neighbor_element == 'P':
                    rdMolTransforms.SetBondLength(conformer, capped_H_neighbor_atom_idx, atom_idx, 1.87)
                else:
                    raise ValueError('Capped atom element not supported!!')

            Chem.SanitizeMol(expanded_root_fragment_H)
            expanded_root_fragment_capped = Chem.AddHs(expanded_root_fragment_H, addCoords=True)

            Chem.AssignStereochemistryFrom3D(expanded_root_fragment_capped)
            _ = Chem.FindMolChiralCenters(expanded_root_fragment_capped, includeUnassigned=True, useLegacyImplementation=False)
            _ = list(Chem.FindPotentialStereo(expanded_root_fragment_capped))
        #############################################################################################

        #############################################################################################
        ## Make whole molecule root atom mapping
        num_expanded_root_fragment_atoms = expanded_root_fragment_capped.GetNumAtoms()
        for root_atom_idx in range(num_expanded_root_fragment_atoms):
            atom = expanded_root_fragment_capped.GetAtomWithIdx(root_atom_idx)
            if atom.HasProp('root_heavy_atom_idx'):
                whole_mol_sdf_atom_idx = atom.GetIntProp('sdf_atom_idx')
                atom.SetIntProp('whole_mol_sdf_atom_idx', whole_mol_sdf_atom_idx)
                whole_mol_atom_idx = whole_mol_sdf_atom_idx - 1
                whole_mol_atom = mol.GetAtomWithIdx(whole_mol_atom_idx)
                whole_mol_atom.SetIntProp('root_atom_idx', root_atom_idx)

        self.atom_typer.assign_atom_types(expanded_root_fragment_capped)
        ComputeGasteigerCharges(expanded_root_fragment_capped)
        utils.assign_atom_properties(expanded_root_fragment_capped)
        #############################################################################################

        #############################################################################################
        ## Assign unidock style sdf properties
        fragment_info_string = ''
        torsion_info_string = ''
        atom_info_string = ''
        torsion_sampling_range_info_string = ''

        num_root_fragment_atoms = expanded_root_fragment_capped.GetNumAtoms()
        fragment_info_string += ' '.join([str(atom_idx+1) for atom_idx in range(num_root_fragment_atoms)])
        fragment_info_string += '\n'

        atom_info_line_format = '{:<3d}{:<10f}{:<2s}\n'
        for atom in expanded_root_fragment_capped.GetAtoms():
            sdf_atom_idx = atom.GetIntProp('sdf_atom_idx')
            charge = atom.GetDoubleProp('charge')
            atom_type = atom.GetProp('atom_type')

            atom_info_string += atom_info_line_format.format(sdf_atom_idx, charge, atom_type)

        expanded_root_fragment_capped.SetProp('fragInfo', fragment_info_string)
        expanded_root_fragment_capped.SetProp('torsionInfo', torsion_info_string)
        expanded_root_fragment_capped.SetProp('atomInfo', atom_info_string)
        expanded_root_fragment_capped.SetProp('torsionSamplingRangeInfo', torsion_sampling_range_info_string)
        #############################################################################################

        self.root_fragment_mol = expanded_root_fragment_capped
        self.root_fragment_smiles = Chem.MolToSmiles(Chem.RemoveHs(self.root_fragment_mol))
        self.mol = mol

        return self.root_fragment_mol, self.mol, self.rotamer_rotatable_bond_info_list, self.rigid_ligand

    def perform_torsion_drives(self):
        num_root_atoms = self.root_fragment_mol.GetNumAtoms()
        root_conformer = self.root_fragment_mol.GetConformer()
        root_positions = root_conformer.GetPositions()
        root_coords = root_positions - np.mean(root_positions, axis=0)
        root_positions_in_pocket = root_coords + self.target_center

        for atom_idx in range(num_root_atoms):
            root_atom_point3D = Point3D(*root_positions_in_pocket[atom_idx, :])
            root_conformer.SetAtomPosition(atom_idx, root_atom_point3D)

        torsion_library_driver = TorsionLibraryDriver(self.root_fragment_mol,
                                                      self.torsion_library_dict,
                                                      self.max_num_root_torsion_enumerations,
                                                      self.n_cpu)

        self.prepared_root_fragment_conf_mol_list = torsion_library_driver.generate_torsion_conformations()
        self.num_root_conformations = len(self.prepared_root_fragment_conf_mol_list)

    def write_root_fragment_sdf_files(self):
        ligand_sdf_base_file_name = os.path.basename(self.ligand_sdf_file_name)
        ligand_file_name_prefix = ligand_sdf_base_file_name.split('.')[0]
        self.ligand_root_sdf_file_name_list = [None] * self.num_root_conformations

        for conf_idx in range(self.num_root_conformations):
            prepared_root_fragment_conf_mol = self.prepared_root_fragment_conf_mol_list[conf_idx]
            ligand_root_sdf_base_file_name = ligand_file_name_prefix + f'_root_conf_{conf_idx}.sdf'
            ligand_root_sdf_file_name = os.path.join(self.working_dir_name, ligand_root_sdf_base_file_name)
            self.ligand_root_sdf_file_name_list[conf_idx] = ligand_root_sdf_file_name
        
        #############################################################################################
        ## Batch parallel writing root sdf files
        raw_num_batches = self.n_cpu
        num_confs_per_batch = int(self.num_root_conformations / raw_num_batches) + 1

        num_batches = 0
        batch_idx_tuple_list = []
        for batch_idx in range(raw_num_batches):
            begin_conf_idx = num_confs_per_batch * batch_idx
            end_conf_idx = num_confs_per_batch * (batch_idx + 1)
            num_batches += 1

            if end_conf_idx >= self.num_root_conformations:
                end_conf_idx = self.num_root_conformations
                batch_idx_tuple = (begin_conf_idx, end_conf_idx)
                batch_idx_tuple_list.append(batch_idx_tuple)
                break
            else:
                batch_idx_tuple = (begin_conf_idx, end_conf_idx)
                batch_idx_tuple_list.append(batch_idx_tuple)

        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        write_root_fragment_pool = Pool(processes=num_batches)
        write_root_fragment_results_list = [None] * num_batches

        for batch_idx in tqdm(range(num_batches)):
            batch_idx_tuple = batch_idx_tuple_list[batch_idx]
            batch_root_sdf_file_name_list = self.ligand_root_sdf_file_name_list[batch_idx_tuple[0]:batch_idx_tuple[1]]
            batch_root_fragment_conf_mol_list = self.prepared_root_fragment_conf_mol_list[batch_idx_tuple[0]:batch_idx_tuple[1]]

            write_root_fragment_results = write_root_fragment_pool.apply_async(batch_write_root_fragment_process,
                                                                               args=(batch_root_sdf_file_name_list,
                                                                                     batch_root_fragment_conf_mol_list))

            write_root_fragment_results_list[batch_idx] = write_root_fragment_results

        write_root_fragment_pool.close()
        write_root_fragment_pool.join()

        write_root_fragment_output_list = [write_root_fragment_results.get() for write_root_fragment_results in write_root_fragment_results_list]
        #############################################################################################
