import os

import numpy as np

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds, rdMolTransforms
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock.unidock_processing.ligand_topology.atom_type import AtomType
from unidock.unidock_processing.ligand_topology.rotatable_bond import RotatableBond
from unidock.unidock_processing.ligand_topology import utils

class UniDockRootBuilder(object):
    def __init__(self,
                 ligand_sdf_file_name,
                 working_dir_name='.'):

        self.ligand_sdf_file_name = ligand_sdf_file_name
        self.working_dir_name = os.path.abspath(working_dir_name)

        ligand_sdf_base_file_name = os.path.basename(self.ligand_sdf_file_name)
        ligand_file_name_prefix = ligand_sdf_base_file_name.split('.')[0]

        ligand_root_sdf_base_file_name = ligand_file_name_prefix + '_root.sdf'
        self.ligand_root_sdf_file_name = os.path.join(self.working_dir_name, ligand_root_sdf_base_file_name)

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
        ## Find fragment as the root node
        num_fragment_atoms_list = [None] * num_fragments
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        root_fragment_idx = np.argmax(num_fragment_atoms_list)
        ########################################################################################

        ########################################################################################
        ## Find rotatable bonds to be removed so that we can construct an expanded root fragment
        root_fragment = splitted_mol_list[root_fragment_idx]
        removed_rotatable_bond_info_list = []
        num_root_atoms = root_fragment.GetNumAtoms()
        for atom_idx in range(num_root_atoms):
            root_atom = root_fragment.GetAtomWithIdx(atom_idx)
            root_atom_idx = int(root_atom.GetProp('sdf_atom_idx')) - 1

            for rotatable_bond_info in rotatable_bond_info_list:
                if root_atom_idx in rotatable_bond_info:
                    if rotatable_bond_info not in removed_rotatable_bond_info_list:
                        removed_rotatable_bond_info_list.append(rotatable_bond_info)

        for removed_rotatable_bond_info in removed_rotatable_bond_info_list:
            rotatable_bond_info_list.remove(removed_rotatable_bond_info)
        ########################################################################################

        ########################################################################################
        ## Construct the expanded root fragment
        refined_rotatable_bond_idx_list = []
        for bond in bond_list:
            bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            if bond_info in rotatable_bond_info_list or bond_info_reversed in rotatable_bond_info_list:
                refined_rotatable_bond_idx_list.append(bond.GetIdx())

        if len(refined_rotatable_bond_idx_list) != 0:
            refined_splitted_mol = FragmentOnBonds(mol, refined_rotatable_bond_idx_list, addDummies=False)
            refined_splitted_mol_list = list(GetMolFrags(refined_splitted_mol, asMols=True, sanitizeFrags=False))
        else:
            refined_splitted_mol_list = [mol]

        num_fragments = len(refined_splitted_mol_list)

        num_fragment_atoms_list = [None] * num_fragments
        for fragment_idx in range(num_fragments):
            fragment = refined_splitted_mol_list[fragment_idx]
            num_atoms = fragment.GetNumAtoms()
            num_fragment_atoms_list[fragment_idx] = num_atoms

        expanded_root_fragment_idx = np.argmax(num_fragment_atoms_list)
        expanded_root_fragment = refined_splitted_mol_list[expanded_root_fragment_idx]
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
            else:
                raise ValueError('Capped atom element not supported!!')

        Chem.SanitizeMol(expanded_root_fragment_H)
        expanded_root_fragment_capped = Chem.AddHs(expanded_root_fragment_H, addCoords=True)
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

        self.root_fragment_mol = expanded_root_fragment_capped
        self.mol = mol
        self.rotatable_bond_info_list = rotatable_bond_info_list

        return self.root_fragment_mol, self.mol, self.rotatable_bond_info_list

    def write_root_fragment_sdf_file(self):
        fragment_info_string = ''
        torsion_info_string = ''
        atom_info_string = ''
        torsion_sampling_range_info_string = ''

        num_root_fragment_atoms = self.root_fragment_mol.GetNumAtoms()
        fragment_info_string += ' '.join([str(atom_idx+1) for atom_idx in range(num_root_fragment_atoms)])
        fragment_info_string += '\n'

        atom_info_line_format = '{:<3d}{:<10f}{:<2s}\n'
        for atom in self.root_fragment_mol.GetAtoms():
            sdf_atom_idx = atom.GetIntProp('sdf_atom_idx')
            charge = atom.GetDoubleProp('charge')
            atom_type = atom.GetProp('vina_atom_type')

            atom_info_string += atom_info_line_format.format(sdf_atom_idx, charge, atom_type)

        self.root_fragment_mol.SetProp('fragInfo', fragment_info_string)
        self.root_fragment_mol.SetProp('torsionInfo', torsion_info_string)
        self.root_fragment_mol.SetProp('atomInfo', atom_info_string)
        self.root_fragment_mol.SetProp('torsionSamplingRangeInfo', torsion_sampling_range_info_string)

        writer = Chem.SDWriter(self.ligand_root_sdf_file_name)
        writer.write(self.root_fragment_mol)
        writer.flush()
        writer.close()
