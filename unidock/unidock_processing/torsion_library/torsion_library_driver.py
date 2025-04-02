from copy import deepcopy

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

from unidock.unidock_processing.torsion_library.utils import get_torsion_atom_idx_tuple, get_torsion_mobile_atom_idx_list, rotate_torsion_angle
from unidock.unidock_processing.torsion_library.torsion_rule_matcher import TorsionRuleMatcher
from unidock.unidock_processing.bounding_volume_hierarchy.utils import construct_oriented_bounding_box_list

class TorsionLibraryDriver(object):
    def __init__(self,
                 mol,
                 rotatable_bond_info_list,
                 torsion_library_dict):

        self.mol = deepcopy(mol)
        self.rotatable_bond_info_list = rotatable_bond_info_list
        self.num_rotatable_bonds = len(self.rotatable_bond_info_list)
        self.torsion_library_dict = torsion_library_dict

    def perform_torsion_matches(self):
        torsion_rule_matcher = TorsionRuleMatcher(self.mol,
                                                  self.rotatable_bond_info_list,
                                                  self.torsion_library_dict)

        torsion_rule_matcher.match_torsion_rules()
        self.matched_torsion_info_dict_list = torsion_rule_matcher.matched_torsion_info_dict_list

        #############################################################################################
        ## Enumerate all torsion values
        self.torsion_atom_idx_nested_list = [None] * self.num_rotatable_bonds
        self.enumerated_torsion_value_nested_list = [None] * self.num_rotatable_bonds
        self.enumerated_torsion_range_nested_list = [None] * self.num_rotatable_bonds
        self.original_torsion_value_list = [None] * self.num_rotatable_bonds

        conformer = self.mol.GetConformer()

        for rotatable_bond_idx in range(self.num_rotatable_bonds):
            rotatable_bond_info = self.rotatable_bond_info_list[rotatable_bond_idx]

            for matched_torsion_info_dict in self.matched_torsion_info_dict_list:
                current_rotatable_bond_info = matched_torsion_info_dict['rotatable_bond_info']
                if current_rotatable_bond_info == rotatable_bond_info:
                    torsion_atom_idx_list = matched_torsion_info_dict['torsion_atom_idx']
                    self.torsion_atom_idx_nested_list[rotatable_bond_idx] = torsion_atom_idx_list
                    self.original_torsion_value_list[rotatable_bond_idx] = Chem.rdMolTransforms.GetDihedralDeg(conformer, *torsion_atom_idx_list)
                    self.enumerated_torsion_value_nested_list[rotatable_bond_idx] = matched_torsion_info_dict['torsion_angle_value']
                    self.enumerated_torsion_range_nested_list[rotatable_bond_idx] = matched_torsion_info_dict['torsion_angle_range']

            if not self.torsion_atom_idx_nested_list[rotatable_bond_idx]:
                torsion_atom_idx_list = get_torsion_atom_idx_tuple(self.mol, rotatable_bond_info)
                self.torsion_atom_idx_nested_list[rotatable_bond_idx] = torsion_atom_idx_list
                self.original_torsion_value_list[rotatable_bond_idx] = Chem.rdMolTransforms.GetDihedralDeg(conformer, *torsion_atom_idx_list)
                self.enumerated_torsion_value_nested_list[rotatable_bond_idx] = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150, 180.0]
                self.enumerated_torsion_range_nested_list[rotatable_bond_idx] = [(-150.0, 180.0)]
        #############################################################################################

    def identify_torsion_mobile_atoms(self, root_atom_idx):
        self.mobile_atom_idx_nested_list = [None] * self.num_rotatable_bonds
        for rotatable_bond_idx in range(self.num_rotatable_bonds):
            rotatable_bond_info = self.rotatable_bond_info_list[rotatable_bond_idx]
            mobile_atom_idx_list = get_torsion_mobile_atom_idx_list(self.mol, rotatable_bond_info, root_atom_idx)
            self.mobile_atom_idx_nested_list[rotatable_bond_idx] = mobile_atom_idx_list

            #########################################################################################
            #########################################################################################
            ## reorder atom idx list so that it starts from root to the rotamer
            torsion_atom_idx_list = self.torsion_atom_idx_nested_list[rotatable_bond_idx]
            torsion_atom_idx_i = torsion_atom_idx_list[0]
            if torsion_atom_idx_i in mobile_atom_idx_list:
                reordered_torsion_atom_idx_list = list(reversed(torsion_atom_idx_list))
            else:
                reordered_torsion_atom_idx_list = torsion_atom_idx_list

            self.torsion_atom_idx_nested_list[rotatable_bond_idx] = reordered_torsion_atom_idx_list
            #########################################################################################
            #########################################################################################

    def generate_obb_for_specified_torsion_sets(self, specified_torsion_value_list, fragment_mol):
        #############################################################################################
        ## Rotate torsions to get specified conformer
        for rotatable_bond_idx in range(self.num_rotatable_bonds):
            torsion_atom_idx_list = self.torsion_atom_idx_nested_list[rotatable_bond_idx]
            torsion_mobile_atom_idx_list = self.mobile_atom_idx_nested_list[rotatable_bond_idx]
            torsion_value = specified_torsion_value_list[rotatable_bond_idx]

            if torsion_value is None:
                torsion_value = self.original_torsion_value_list[rotatable_bond_idx]

            rotate_torsion_angle(self.mol, torsion_atom_idx_list, torsion_mobile_atom_idx_list, torsion_value)

        coords_array = self.mol.GetConformer().GetPositions()
        #############################################################################################

        fragment_mol_conformer = fragment_mol.GetConformer()
        num_fragment_atoms = fragment_mol.GetNumAtoms()
        for fragment_atom_idx in range(num_fragment_atoms):
            atom = fragment_mol.GetAtomWithIdx(fragment_atom_idx)
            atom_idx = atom.GetIntProp('internal_atom_idx')
            atom_coords = coords_array[atom_idx, :]
            atom_point_3D = Point3D(atom_coords[0], atom_coords[1], atom_coords[2])
            fragment_mol_conformer.SetAtomPosition(fragment_atom_idx, atom_point_3D)

        fragment_obb_list, fragment_obb_info_dict_list = construct_oriented_bounding_box_list(fragment_mol)

        return fragment_obb_list, fragment_obb_info_dict_list
