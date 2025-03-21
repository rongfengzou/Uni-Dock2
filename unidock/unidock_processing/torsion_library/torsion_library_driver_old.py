import itertools
from copy import deepcopy
from tqdm.notebook import tqdm

from rdkit import Chem

from unidock.unidock_processing.ligand_preparation.conformation_processing import check_conformer_vdw_clash
from unidock.unidock_processing.ligand_topology.generic_rotatable_bond import GenericRotatableBond
from unidock.unidock_processing.torsion_library.utils import get_torsion_atom_idx_tuple
from unidock.unidock_processing.torsion_library.torsion_rule_matcher import TorsionRuleMatcher

def torsion_conformer_process(conf_mol, torsion_atom_idx_nested_list, torsion_value_list):
    conformer = conf_mol.GetConformer()
    num_rotatable_bonds = len(torsion_atom_idx_nested_list)

    for rotatable_bond_idx in range(num_rotatable_bonds):
        torsion_atom_idx_list = torsion_atom_idx_nested_list[rotatable_bond_idx]
        torsion_value = torsion_value_list[rotatable_bond_idx]
        Chem.rdMolTransforms.SetDihedralDeg(conformer, *torsion_atom_idx_list, torsion_value)

class TorsionLibraryDriver(object):
    def __init__(self,
                 mol,
                 torsion_library_dict,
                 max_num_rotatable_bonds=5,
                 n_cpu=16):

        self.mol = mol
        self.rotatable_bond_finder = GenericRotatableBond()
        self.rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(self.mol)
        self.num_rotatable_bonds = len(self.rotatable_bond_info_list)
        self.torsion_library_dict = torsion_library_dict
        self.max_num_rotatable_bonds = max_num_rotatable_bonds
        self.n_cpu = n_cpu

    def generate_torsion_conformations(self):
        torsion_rule_matcher = TorsionRuleMatcher(self.mol,
                                                  self.rotatable_bond_info_list,
                                                  self.torsion_library_dict)

        torsion_rule_matcher.match_torsion_rules()
        self.matched_torsion_info_dict_list = torsion_rule_matcher.matched_torsion_info_dict_list

        #############################################################################################
        ## Enumerate all torsion values

        #############################################################################################
        ## FIX ME: Escape for case with too many rotatable bonds
        if self.max_num_rotatable_bonds < self.num_rotatable_bonds:
            print(f'Originally {self.num_rotatable_bonds} rotatable bonds. Only select top {self.max_num_rotatable_bonds}.')
            self.num_rotatable_bonds = self.max_num_rotatable_bonds
        #############################################################################################

        self.torsion_atom_idx_nested_list = [None] * self.num_rotatable_bonds
        self.torsion_value_nested_list = [None] * self.num_rotatable_bonds

        for rotatable_bond_idx in range(self.num_rotatable_bonds):
            rotatable_bond_info = self.rotatable_bond_info_list[rotatable_bond_idx]

            for matched_torsion_info_dict in self.matched_torsion_info_dict_list:
                current_rotatable_bond_info = matched_torsion_info_dict['rotatable_bond_info']
                if current_rotatable_bond_info == rotatable_bond_info:
                    self.torsion_atom_idx_nested_list[rotatable_bond_idx] = matched_torsion_info_dict['torsion_atom_idx']
                    self.torsion_value_nested_list[rotatable_bond_idx] = matched_torsion_info_dict['torsion_angle_value']

            if not self.torsion_atom_idx_nested_list[rotatable_bond_idx]:
                self.torsion_atom_idx_nested_list[rotatable_bond_idx] = get_torsion_atom_idx_tuple(self.mol, rotatable_bond_info)
                self.torsion_value_nested_list[rotatable_bond_idx] = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150, 180.0]
        #############################################################################################

        #############################################################################################
        ## Enumerate torsion conformations and discard vdw clash conformers
        torsion_value_enumeration_list = list(itertools.product(*self.torsion_value_nested_list))
        num_conformations = len(torsion_value_enumeration_list)

        enumerated_conf_mol_list = []

        for conf_idx in tqdm(range(num_conformations)):
            conf_mol = deepcopy(self.mol)
            torsion_value_list = torsion_value_enumeration_list[conf_idx]

            torsion_conformer_process(conf_mol, self.torsion_atom_idx_nested_list, torsion_value_list)

            if not check_conformer_vdw_clash(conf_mol):
                enumerated_conf_mol_list.append(conf_mol)
        #############################################################################################

        return enumerated_conf_mol_list
