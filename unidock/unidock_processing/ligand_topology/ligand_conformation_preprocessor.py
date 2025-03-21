import os
from copy import deepcopy

import multiprocess as mp
from multiprocess.pool import Pool

from rdkit import Chem
from rdkit.Chem import rdMolTransforms, TorsionFingerprints, ChemicalForceFields
from rdkit.Chem.PropertyMol import PropertyMol

def sample_amide_torsions(mol, covalent_ligand=False):
    ###################################################################################################
    ## get covalent atom indices
    if covalent_ligand:
        covalent_atom_idx_string = mol.GetProp('covalent_atom_indices')
        covalent_atom_idx_string_list = covalent_atom_idx_string.split(',')
        covalent_atom_idx_list = [int(covalent_atom_idx_string) for covalent_atom_idx_string in covalent_atom_idx_string_list]
    ###################################################################################################

    amide_pattern_mol = Chem.MolFromSmarts('[$(C=O);!R]-[$(NC=O);D3;H0;0]')
    match_atom_idx_tuple_list = list(mol.GetSubstructMatches(amide_pattern_mol))
    num_amide_torsions = len(match_atom_idx_tuple_list)

    if num_amide_torsions == 0:
        return [mol]

    amide_torsion_info_list = []
    non_amide_torsion_info_list = []
    refined_mol_list = []

    raw_torsion_info_list = TorsionFingerprints.CalculateTorsionLists(mol)[0]
    for raw_torsion_info in raw_torsion_info_list:
        torsion_atom_idx_tuple = raw_torsion_info[0][0]
        rotatable_bond_atom_idx_tuple = (torsion_atom_idx_tuple[1], torsion_atom_idx_tuple[2])
        if rotatable_bond_atom_idx_tuple in match_atom_idx_tuple_list or rotatable_bond_atom_idx_tuple[::-1] in match_atom_idx_tuple_list:
            amide_torsion_info_list.append(raw_torsion_info)
        else:
            non_amide_torsion_info_list.append(raw_torsion_info)

    num_non_amide_torsions = len(non_amide_torsion_info_list)
    if len(amide_torsion_info_list) != num_amide_torsions:
        raise ValueError('Bugs in amide torsion recognitions!!')

    for amide_torsion_idx in range(num_amide_torsions):
        amide_torsion_info = amide_torsion_info_list[amide_torsion_idx]
        amide_torsion_atom_idx_tuple = amide_torsion_info[0][0]

        for tuned_amide_torsion_angle in [0.0, 180.0]:
            current_mol = deepcopy(mol)
            current_conformer = current_mol.GetConformer()
            rdMolTransforms.SetDihedralDeg(current_conformer, *amide_torsion_atom_idx_tuple, tuned_amide_torsion_angle)
            ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(current_mol, 'MMFF94s')
            ff = ChemicalForceFields.MMFFGetMoleculeForceField(current_mol, ff_property)

            ###################################################################################################
            ## covalent atoms should be constrained
            if covalent_ligand:
                for covalent_atom_idx in covalent_atom_idx_list:
                    ff.MMFFAddPositionConstraint(covalent_atom_idx, 0.0, 1000.0)
            ###################################################################################################

            ff.MMFFAddTorsionConstraint(*amide_torsion_atom_idx_tuple, relative=True, minDihedralDeg=-30.0, maxDihedralDeg=30.0, forceConstant=10.0)
            ff.Initialize()

            max_minimize_iteration = 5
            for _ in range(max_minimize_iteration):
                minimize_seed = ff.Minimize(forceTol=1e-3, energyTol=1e-4)
                if minimize_seed == 0:
                    break

            if ff.CalcEnergy() <= 350.0:
                refined_mol_list.append(current_mol)
            else:
                # try to iteratively tune non-amide torsion angles to sample conformations
                raw_non_amide_torsion_value_list = TorsionFingerprints.CalculateTorsionAngles(current_mol, non_amide_torsion_info_list, [])
                for non_amide_torsion_idx in range(num_non_amide_torsions):
                    torsion_atom_idx_tuple = non_amide_torsion_info_list[non_amide_torsion_idx][0][0]
                    torsion_initial_value = raw_non_amide_torsion_value_list[non_amide_torsion_idx][0][0]

                    torsion_refined_value_positive = torsion_initial_value + 60.0
                    if torsion_refined_value_positive >= 180.0:
                        torsion_refined_value_positive -= 360.0

                    torsion_refined_value_negative = torsion_initial_value - 60.0
                    if torsion_refined_value_negative <= -180.0:
                        torsion_refined_value_negative += 360.0

                    ff.MMFFAddTorsionConstraint(*torsion_atom_idx_tuple, relative=True, minDihedralDeg=-30.0, maxDihedralDeg=30.0, forceConstant=10.0)
                    ff.Initialize()

                    # start positive torsion tuning test
                    rdMolTransforms.SetDihedralDeg(current_conformer, *torsion_atom_idx_tuple, torsion_refined_value_positive)
                    max_minimize_iteration = 5
                    for _ in range(max_minimize_iteration):
                        minimize_seed = ff.Minimize(forceTol=1e-3, energyTol=1e-4)
                        if minimize_seed == 0:
                            break

                    if ff.CalcEnergy() <= 350.0:
                        refined_mol_list.append(current_mol)
                        break

                    # start negative torsion tuning test
                    rdMolTransforms.SetDihedralDeg(current_conformer, *torsion_atom_idx_tuple, torsion_refined_value_negative)
                    max_minimize_iteration = 5
                    for _ in range(max_minimize_iteration):
                        minimize_seed = ff.Minimize(forceTol=1e-3, energyTol=1e-4)
                        if minimize_seed == 0:
                            break

                    if ff.CalcEnergy() <= 350.0:
                        refined_mol_list.append(current_mol)
                        break

    if len(refined_mol_list) == 0:
        return [mol]
    else:
        return refined_mol_list

def enumerate_refine_ligand_conformations_process(sdf_file_name,
                                                  refine_amide_torsions,
                                                  covalent_ligand,
                                                  working_dir_name,
                                                  refined_mol_proxy_list,
                                                  refined_sdf_file_name_proxy_list,
                                                  mol_idx):

    mol_iterator = Chem.SDMolSupplier(sdf_file_name, removeHs=False)
    num_conformations = len(mol_iterator)
    enumerated_refined_mol_list = []

    for conf_idx in range(num_conformations):
        mol = mol_iterator[conf_idx]

        if refine_amide_torsions:
            ###############################################################################################################
            # convert dummy atoms to hydrogen (used in covalent docking)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
                    atom.SetIntProp('dummy_atom', 1)
            ###############################################################################################################

            refined_mol_list = sample_amide_torsions(mol, covalent_ligand)

            ###############################################################################################################
            # convert labeled hydrogen atoms back to dummy atoms (used in covalent docking)
            for refined_mol in refined_mol_list:
                for atom in refined_mol.GetAtoms():
                    if atom.HasProp('dummy_atom'):
                        atom.SetAtomicNum(0)
            ###############################################################################################################

        else:
            refined_mol_list = [mol]

        enumerated_refined_mol_list.extend(refined_mol_list)

    sdf_base_file_name_prefix = os.path.basename(sdf_file_name).split('.')[0]
    num_enumerated_conformations = len(enumerated_refined_mol_list)
    enumerated_refined_sdf_file_name_list = [None] * num_enumerated_conformations

    for enumerated_conf_idx in range(num_enumerated_conformations):
        enumerated_refined_mol = enumerated_refined_mol_list[enumerated_conf_idx]
        enumerated_refined_mol.ClearProp('conformer_idx')
        enumerated_refined_mol.ClearProp('conformer_energy')

        sdf_base_file_name = sdf_base_file_name_prefix + '_split_' + str(enumerated_conf_idx) + '.sdf'
        sdf_file_name = os.path.join(working_dir_name, sdf_base_file_name)
        enumerated_refined_sdf_file_name_list[enumerated_conf_idx] = sdf_file_name

        sdf_writer = Chem.SDWriter(sdf_file_name)
        sdf_writer.write(enumerated_refined_mol)
        sdf_writer.flush()
        sdf_writer.close()

        ###############################################################################################################
        ###############################################################################################################
        ## Deal with rdkit V3000 sdf file reader that would force COLLECTION BLOCK to be V3000.
        with open(sdf_file_name, 'r') as f:
            sdf_line_list = f.readlines()

        num_sdf_lines = len(sdf_line_list)
        if sdf_line_list[3].strip().split()[-1] == 'V3000':
            for line_idx in range(num_sdf_lines):
                sdf_line = sdf_line_list[line_idx]

                if sdf_line.startswith('M  V30 BEGIN COLLECTION'):
                    v3000_collection_begin_line_idx = line_idx
                elif sdf_line.startswith('M  V30 END COLLECTION'):
                    v3000_collection_end_line_idx = line_idx

            cleaned_sdf_line_list = sdf_line_list[:v3000_collection_begin_line_idx] + sdf_line_list[v3000_collection_end_line_idx+1:]

            with open(sdf_file_name, 'w') as f:
                for cleaned_sdf_line in cleaned_sdf_line_list:
                    f.write(cleaned_sdf_line)

        ###############################################################################################################
        ###############################################################################################################

    refined_mol_proxy_list[mol_idx] = [PropertyMol(enumerated_refined_mol) for enumerated_refined_mol in enumerated_refined_mol_list]
    refined_sdf_file_name_proxy_list[mol_idx] = enumerated_refined_sdf_file_name_list

    return True

class LigandConformationPreprocessor(object):
    def __init__(self,
                 ligand_sdf_file_name_list,
                 refine_amide_torsions=False,
                 covalent_ligand=False,
                 n_cpu=16,
                 working_dir_name='.'):

        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.refine_amide_torsions = refine_amide_torsions
        self.covalent_ligand = covalent_ligand
        self.n_cpu = n_cpu
        self.num_molecules = len(self.ligand_sdf_file_name_list)
        self.working_dir_name = os.path.abspath(working_dir_name)

    def generate_refined_ligand_sdf_files(self):
        self.extended_ligand_mol_list = []
        self.extended_ligand_sdf_file_name_list = []

        manager = mp.Manager()
        refined_mol_proxy_list = manager.list()
        refined_sdf_file_name_proxy_list = manager.list()
        refined_mol_proxy_list.extend([None] * self.num_molecules)
        refined_sdf_file_name_proxy_list.extend([None] * self.num_molecules)
        conformation_refinement_results_list = [None] * self.num_molecules
        ligand_conformation_refinement_pool = Pool(processes=self.n_cpu)

        for mol_idx in range(self.num_molecules):
            sdf_file_name = self.ligand_sdf_file_name_list[mol_idx]
            conformation_refinement_results = ligand_conformation_refinement_pool.apply_async(enumerate_refine_ligand_conformations_process,
                                                                                              args=(sdf_file_name,
                                                                                                    self.refine_amide_torsions,
                                                                                                    self.covalent_ligand,
                                                                                                    self.working_dir_name,
                                                                                                    refined_mol_proxy_list,
                                                                                                    refined_sdf_file_name_proxy_list,
                                                                                                    mol_idx))

            conformation_refinement_results_list[mol_idx] = conformation_refinement_results

        ligand_conformation_refinement_pool.close()
        ligand_conformation_refinement_pool.join()

        conformation_refinement_results_list = [conformation_refinement_results.get() for conformation_refinement_results in conformation_refinement_results_list]
        refined_mol_list = list(refined_mol_proxy_list)
        refined_sdf_file_name_list = list(refined_sdf_file_name_proxy_list)

        for mol_idx in range(self.num_molecules):
            current_refined_mol_list = refined_mol_list[mol_idx]
            current_refined_sdf_file_name_list = refined_sdf_file_name_list[mol_idx]
            self.extended_ligand_mol_list.extend(current_refined_mol_list)
            self.extended_ligand_sdf_file_name_list.extend(current_refined_sdf_file_name_list)
