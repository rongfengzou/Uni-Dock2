import os
from multiprocess.pool import Pool

from unidock.unidock_processing.ligand_topology.autodock_topology_builder import AutoDockTopologyBuilder

def prepare_ligand_torsion_tree_file_process(ligand_sdf_file_name,
                                             covalent_ligand,
                                             template_docking,
                                             reference_sdf_file_name,
                                             core_atom_mapping_dict,
                                             use_torsion_tree_sdf,
                                             working_dir_name):

    autodock_topology_builder = AutoDockTopologyBuilder(ligand_sdf_file_name,
                                                        covalent_ligand=covalent_ligand,
                                                        template_docking=template_docking,
                                                        reference_sdf_file_name=reference_sdf_file_name,
                                                        core_atom_mapping_dict=core_atom_mapping_dict,
                                                        working_dir_name=working_dir_name)

    autodock_topology_builder.build_molecular_graph()
    autodock_topology_builder.write_pdbqt_file()

    if template_docking:
        autodock_topology_builder.write_constraint_bpf_file()

    if use_torsion_tree_sdf:
        autodock_topology_builder.write_torsion_tree_sdf_file()

        return autodock_topology_builder.ligand_pdbqt_file_name, autodock_topology_builder.ligand_torsion_tree_sdf_file_name

    else:
        return autodock_topology_builder.ligand_pdbqt_file_name

class AutoDockTorsionTreeGenerator(object):
    def __init__(self,
                 ligand_sdf_file_name_list,
                 covalent_ligand=False,
                 template_docking=False,
                 reference_sdf_file_name=None,
                 core_atom_mapping_dict_list=None,
                 generate_torsion_tree_sdf=False,
                 n_cpu=16,
                 working_dir_name='.'):

        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.covalent_ligand = covalent_ligand
        self.template_docking = template_docking
        self.reference_sdf_file_name = reference_sdf_file_name
        self.core_atom_mapping_dict_list = core_atom_mapping_dict_list
        self.generate_torsion_tree_sdf = generate_torsion_tree_sdf
        self.n_cpu = n_cpu
        self.num_molecules = len(self.ligand_sdf_file_name_list)
        self.working_dir_name = os.path.abspath(working_dir_name)

    def generate_ligand_torsion_tree_files(self):
        self.ligand_pdbqt_file_name_list = [None] * self.num_molecules

        if self.generate_torsion_tree_sdf:
            self.ligand_torsion_tree_sdf_file_name_list = [None] * self.num_molecules

        torsion_tree_results_list = [None] * self.num_molecules
        ligand_torsion_tree_preparation_pool = Pool(processes=self.n_cpu)

        for mol_idx in range(self.num_molecules):
            ligand_sdf_file_name = self.ligand_sdf_file_name_list[mol_idx]
            core_atom_mapping_dict = self.core_atom_mapping_dict_list[mol_idx]
            torsion_tree_results = ligand_torsion_tree_preparation_pool.apply_async(prepare_ligand_torsion_tree_file_process,
                                                                                    args=(ligand_sdf_file_name,
                                                                                          self.covalent_ligand,
                                                                                          self.template_docking,
                                                                                          self.reference_sdf_file_name,
                                                                                          core_atom_mapping_dict,
                                                                                          self.generate_torsion_tree_sdf,
                                                                                          self.working_dir_name))

            torsion_tree_results_list[mol_idx] = torsion_tree_results

        ligand_torsion_tree_preparation_pool.close()
        ligand_torsion_tree_preparation_pool.join()

        ligand_torsion_tree_info_nested_list = [torsion_tree_results.get() for torsion_tree_results in torsion_tree_results_list]

        if self.generate_torsion_tree_sdf:
            for mol_idx in range(self.num_molecules):
                ligand_torsion_tree_info_list = ligand_torsion_tree_info_nested_list[mol_idx]
                self.ligand_pdbqt_file_name_list[mol_idx] = ligand_torsion_tree_info_list[0]
                self.ligand_torsion_tree_sdf_file_name_list[mol_idx] = ligand_torsion_tree_info_list[1]

        else:
            self.ligand_pdbqt_file_name_list = ligand_torsion_tree_info_nested_list
