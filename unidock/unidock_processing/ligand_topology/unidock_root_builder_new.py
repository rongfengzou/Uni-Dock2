import os
from copy import deepcopy

import numpy as np

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds, rdMolTransforms
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Geometry import Point3D

from unidock.unidock_processing.ligand_topology.atom_type import AtomType
from unidock.unidock_processing.ligand_topology.rotatable_bond import RotatableBond
from unidock.unidock_processing.ligand_topology import utils

from unidock.unidock_processing.ligand_preparation.ligand_conformation_generator_conforge import LigandConformationGenerator

class UniDockRootBuilder(object):
    def __init__(self,
                 ligand_sdf_file_name,
                 target_center=(0.0, 0.0, 0.0),
                 n_cpu=16,
                 working_dir_name='.'):

        self.ligand_sdf_file_name = ligand_sdf_file_name
        self.target_center = np.array(target_center)
        self.n_cpu = n_cpu
        self.working_dir_name = os.path.abspath(working_dir_name)

        self.atom_typer = AtomType()
        self.rotatable_bond_finder = RotatableBond()

    def build_root_fragment(self):
        mol = Chem.SDMolSupplier(self.ligand_sdf_file_name, removeHs=False)[0]

        self.atom_typer.assign_atom_types(mol)
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(mol)

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

        ########################################################################################
        ## Find end group fragments (end group serve as the rotamer group of this mol, that means it connect to only one rotatable bond)
        end_group_fragment_list = []
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            rotatable_bond_count = 0
            for atom in fragment.GetAtoms():
                internal_atom_idx = atom.GetIntProp('internal_atom_idx')
                for rotatable_bond_info in rotatable_bond_info_list:
                    if internal_atom_idx in rotatable_bond_info:
                        rotatable_bond_count += 1

            if rotatable_bond_count == 1:
                end_group_fragment_list.append(fragment)
        ########################################################################################

        ########################################################################################
        ## Find selected rotatable bonds and label rotamer group atoms
        num_end_groups = len(end_group_fragment_list)
        if num_end_groups == 0:
            self.rigid_ligand = True
            selected_rotatable_bond_info_list = []
        elif num_end_groups == num_fragments:
            self.rigid_ligand = False
            if num_end_groups != 2:
                raise ValueError('Not exactly 2 fragments in this molecule!! Please look at this molecule and check algorithm carefully!!')

            selected_rotatable_bond_info_list = rotatable_bond_info_list

            num_fragment_atoms_list = [None] * num_end_groups
            for fragment_idx in range(num_end_groups):
                fragment = end_group_fragment_list[fragment_idx]
                num_atoms = fragment.GetNumAtoms()
                num_fragment_atoms_list[fragment_idx] = num_atoms

            end_group_fragment_idx = np.argmin(num_fragment_atoms_list)
            end_group_fragment = end_group_fragment_list[end_group_fragment_idx]
            for atom in end_group_fragment.GetAtoms():
                if atom.GetSymbol() != 'H':
                    internal_atom_idx = atom.GetIntProp('internal_atom_idx')
                    whole_mol_atom = mol.GetAtomWithIdx(internal_atom_idx)
                    whole_mol_atom.SetBoolProp('rotamer_group', True)

        else:
            self.rigid_ligand = False
            selected_rotatable_bond_info_set = set()
            for fragment in end_group_fragment_list:
                found_rotatable_bond = False
                for atom in fragment.GetAtoms():
                    internal_atom_idx = atom.GetIntProp('internal_atom_idx')
                    for rotatable_bond_info in rotatable_bond_info_list:
                        if internal_atom_idx in rotatable_bond_info:
                            selected_rotatable_bond_info_set.add(rotatable_bond_info)
                            found_rotatable_bond = True
                            break

                    if found_rotatable_bond:
                        break

            selected_rotatable_bond_info_list = list(selected_rotatable_bond_info_set)

            for end_group_fragment in end_group_fragment_list:
                for atom in end_group_fragment.GetAtoms():
                    if atom.GetSymbol() != 'H':
                        internal_atom_idx = atom.GetIntProp('internal_atom_idx')
                        whole_mol_atom = mol.GetAtomWithIdx(internal_atom_idx)
                        whole_mol_atom.SetBoolProp('rotamer_group', True)

        rotatable_bond_info_list = selected_rotatable_bond_info_list
        ########################################################################################

        ########################################################################################
        ## Redo fragmentization and get the expanded root fragment
        if self.rigid_ligand:
            expanded_root_fragment = mol
            expanded_root_fragment_capped = mol
        else:
            refined_rotatable_bond_idx_list = []
            for bond in bond_list:
                bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                if bond_info in rotatable_bond_info_list or bond_info_reversed in rotatable_bond_info_list:
                    refined_rotatable_bond_idx_list.append(bond.GetIdx())

            if len(refined_rotatable_bond_idx_list) > 0:
                refined_splitted_mol = FragmentOnBonds(mol, refined_rotatable_bond_idx_list, addDummies=False)
                refined_splitted_mol_list = list(GetMolFrags(refined_splitted_mol, asMols=True, sanitizeFrags=False))
            else:
                raise ValueError('Bugs in end group and rotatable bond assignment codes!!')

            num_fragments = len(refined_splitted_mol_list)

            found_root_fragment_idx = False
            for fragment_idx in range(num_fragments):
                fragment = refined_splitted_mol_list[fragment_idx]
                for atom in fragment.GetAtoms():
                    if atom.GetSymbol() != 'H' and not atom.HasProp('rotamer_group'):
                        root_fragment_idx = fragment_idx
                        found_root_fragment_idx = True
                        break

                if found_root_fragment_idx:
                    break

            expanded_root_fragment = refined_splitted_mol_list[root_fragment_idx]
        ########################################################################################

        ########################################################################################
        ## Assign heavy atom idx for root fragments for later whole mol atom mapping
        heavy_atom_idx = 0
        for atom in expanded_root_fragment.GetAtoms():
            if atom.GetSymbol() != 'H':
                atom.SetIntProp('root_heavy_atom_idx', heavy_atom_idx)
                heavy_atom_idx += 1
        ########################################################################################

        ########################################################################################
        ## Add capped fragments on root fragment
        if not self.rigid_ligand:
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
        ########################################################################################

        ########################################################################################
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
        ########################################################################################

        ########################################################################################
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
        ########################################################################################

        self.root_fragment_mol = expanded_root_fragment_capped
        self.root_fragment_smiles = Chem.MolToSmiles(Chem.RemoveHs(self.root_fragment_mol))
        self.mol = mol
        self.rotatable_bond_info_list = rotatable_bond_info_list

        return self.root_fragment_mol, self.mol, self.rotatable_bond_info_list, self.rigid_ligand

    def generate_conformations_for_root(self):
        ligand_conformation_generator = LigandConformationGenerator([self.root_fragment_smiles],
                                                                    ligand_molecule_name_list=['ligand_root_fragment'],
                                                                    core_sdf_file_name=None,
                                                                    covalent_ligand=False,
                                                                    n_cpu=self.n_cpu,
                                                                    max_num_confs_per_isomer_conf_gen=1000,
                                                                    max_num_confs_per_isomer_cluster=1000,
                                                                    minimize_conformers=False,
                                                                    remove_twisted_six_rings=False,
                                                                    keep_by_rotatable_bonds=True,
                                                                    extend_timeout_for_macrocycles=True,
                                                                    conf_gen_time_limit=2*60,
                                                                    max_stereo_num=32,
                                                                    working_dir_name=self.working_dir_name)

        _, _ = ligand_conformation_generator.run_conformation_generation()
        prepared_root_mol_list = ligand_conformation_generator.conformer_info_df.loc[:, 'ROMol'].values.tolist()
        self.num_root_conformations = len(prepared_root_mol_list)

        prepared_root_confgen_sdf_file_name = os.path.join(self.working_dir_name, 'ligand_confs_prepared.sdf')
        writer = Chem.SDWriter(prepared_root_confgen_sdf_file_name)
        for conf_idx in range(self.num_root_conformations):
            prepared_root_mol = prepared_root_mol_list[conf_idx]
            writer.write(prepared_root_mol)
            writer.flush()

        writer.close()

        num_root_atoms = self.root_fragment_mol.GetNumAtoms()
        self.prepared_root_fragment_conf_mol_list = [None] * self.num_root_conformations

        root_fragment_mol_no_H = Chem.RemoveHs(self.root_fragment_mol)
        for conf_idx in range(self.num_root_conformations):
            current_root_fragment_mol_no_H = deepcopy(root_fragment_mol_no_H)
            current_root_fragment_mol = deepcopy(self.root_fragment_mol)
            current_root_fragment_conformer = current_root_fragment_mol.GetConformer()

            prepared_root_mol = prepared_root_mol_list[conf_idx]
            num_prepared_root_atoms = prepared_root_mol.GetNumAtoms()
            for atom_idx in range(num_prepared_root_atoms):
                atom = prepared_root_mol.GetAtomWithIdx(atom_idx)
                atom.SetIntProp('internal_atom_idx', atom_idx)

            prepared_root_mol_no_H = Chem.RemoveHs(prepared_root_mol)

            root_conf_heavy_atom_mapping = list(current_root_fragment_mol_no_H.GetSubstructMatches(prepared_root_mol_no_H))[0]
            heavy_atom_mapping_dict = {root_atom_idx: prepared_root_atom_idx for prepared_root_atom_idx, root_atom_idx in enumerate(root_conf_heavy_atom_mapping)}
            full_atom_mapping_dict = utils.recover_full_atom_mapping_from_heavy_atoms(current_root_fragment_mol,
                                                                                      current_root_fragment_mol_no_H,
                                                                                      prepared_root_mol,
                                                                                      prepared_root_mol_no_H,
                                                                                      heavy_atom_mapping_dict)

            num_mapped_atoms = len(full_atom_mapping_dict)
            if num_mapped_atoms != num_root_atoms:
                raise ValueError('Possible bugs in root conformation atom mappings!!')

            prepared_root_conf = prepared_root_mol.GetConformer()
            prepared_root_positions = prepared_root_conf.GetPositions()
            prepared_root_coords = prepared_root_positions - np.mean(prepared_root_positions, axis=0)
            prepared_root_positions_in_pocket = prepared_root_coords + self.target_center

            for mapped_root_atom_idx in full_atom_mapping_dict:
                mapped_prepared_root_atom_idx = full_atom_mapping_dict[mapped_root_atom_idx]
                prepared_atom_point3D = Point3D(*prepared_root_positions_in_pocket[mapped_prepared_root_atom_idx, :])
                current_root_fragment_conformer.SetAtomPosition(mapped_root_atom_idx, prepared_atom_point3D)

            self.prepared_root_fragment_conf_mol_list[conf_idx] = current_root_fragment_mol

    def write_root_fragment_sdf_files(self):
        ligand_sdf_base_file_name = os.path.basename(self.ligand_sdf_file_name)
        ligand_file_name_prefix = ligand_sdf_base_file_name.split('.')[0]

        self.ligand_root_sdf_file_name_list = [None] * self.num_root_conformations
        for conf_idx in range(self.num_root_conformations):
            prepared_root_fragment_conf_mol = self.prepared_root_fragment_conf_mol_list[conf_idx]
            ligand_root_sdf_base_file_name = ligand_file_name_prefix + f'_root_conf_{conf_idx}.sdf'
            ligand_root_sdf_file_name = os.path.join(self.working_dir_name, ligand_root_sdf_base_file_name)
            self.ligand_root_sdf_file_name_list[conf_idx] = ligand_root_sdf_file_name

            writer = Chem.SDWriter(ligand_root_sdf_file_name)
            writer.write(prepared_root_fragment_conf_mol)
            writer.flush()
            writer.close()
