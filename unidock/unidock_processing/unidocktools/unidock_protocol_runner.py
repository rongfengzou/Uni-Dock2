from typing import List, Tuple, Dict, Optional, Any
import os
import json
import yaml
from shutil import rmtree

from rdkit import Chem

from unidock_engine.api.python.pipeline import run_docking_pipeline
from unidock_processing.unidocktools.unidock_receptor_topology_builder import UnidockReceptorTopologyBuilder
from unidock_processing.unidocktools.unidock_ligand_topology_builder import UnidockLigandTopologyBuilder
from unidock_processing.unidocktools.unidock_ligand_pose_writer import UnidockLigandPoseWriter
from unidock_processing.ligand_topology import utils

class UnidockProtocolRunner(object):
    def __init__(self,
                 receptor_file_name,
                 ligand_sdf_file_name_list,
                 target_center,
                 option_yaml_file_name=None):

        self.receptor_file_name = receptor_file_name
        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.target_center = target_center

        if option_yaml_file_name is not None:
            self.option_yaml_file_name = option_yaml_file_name
        else:
            self.option_yaml_file_name = os.path.join(os.path.dirname(__file__), 'data', 'unidock_option_template.yaml')

        with open(self.option_yaml_file_name, 'r') as option_yaml_file:
            self.unidock2_option_dict = yaml.safe_load(option_yaml_file)

        self.template_docking = self.unidock2_option_dict['Preprocessing']['template_docking']
        self.reference_sdf_file_name = self.unidock2_option_dict['Preprocessing']['reference_sdf_file_name']
        self.covalent_ligand = self.unidock2_option_dict['Preprocessing']['covalent_ligand']
        self.covalent_residue_atom_info_list = self.unidock2_option_dict['Preprocessing']['covalent_residue_atom_info_list']
        self.preserve_receptor_hydrogen = self.unidock2_option_dict['Preprocessing']['preserve_receptor_hydrogen']
        self.remove_temp_files = self.unidock2_option_dict['Preprocessing']['remove_temp_files']

        self.working_dir_name = os.path.abspath(self.unidock2_option_dict['Preprocessing']['working_dir_name'])
        self.unidock2_output_working_dir_name = os.path.join(self.working_dir_name, 'unidock2_output')

        if os.path.isdir(self.unidock2_output_working_dir_name):
            rmtree(self.unidock2_output_working_dir_name, ignore_errors=True)
            os.mkdir(self.unidock2_output_working_dir_name)
        else:
            os.mkdir(self.unidock2_output_working_dir_name)

        raw_core_atom_mapping_dict_list = self.unidock2_option_dict['Preprocessing']['core_atom_mapping_dict_list']

        if raw_core_atom_mapping_dict_list is None:
            self.core_atom_mapping_dict_list = None
        else:
            num_molecules = len(raw_core_atom_mapping_dict_list)
            self.core_atom_mapping_dict_list = [None] * num_molecules

            for mol_idx in range(num_molecules):
                raw_core_atom_mapping_dict = raw_core_atom_mapping_dict_list[mol_idx]
                if raw_core_atom_mapping_dict is None:
                    self.core_atom_mapping_dict_list[mol_idx] = None
                else:
                    core_atom_mapping_dict = {int(reference_atom_idx): int(query_atom_idx) for reference_atom_idx, query_atom_idx in raw_core_atom_mapping_dict.items()}
                    self.core_atom_mapping_dict_list[mol_idx] = core_atom_mapping_dict

        if self.template_docking and self.target_center == (0.0, 0.0, 0.0):
            reference_mol = Chem.SDMolSupplier(self.reference_sdf_file_name, removeHs=True)[0]
            self.target_center = tuple(utils.calculate_center_of_mass(reference_mol))

    def run_unidock_protocol(self):
        ## prepare receptor
        unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
            self.receptor_file_name,
            prepared_hydrogen=self.preserve_receptor_hydrogen,
            covalent_residue_atom_info_list=self.covalent_residue_atom_info_list,
            working_dir_name=self.working_dir_name
        )

        unidock_receptor_topology_builder.generate_receptor_topology()
        unidock_receptor_topology_builder.get_summary_receptor_info_dict()

        ## prepare ligands input
        unidock_ligand_topology_builder = UnidockLigandTopologyBuilder(
            self.ligand_sdf_file_name_list,
            covalent_ligand=self.covalent_ligand,
            template_docking=self.template_docking,
            reference_sdf_file_name=self.reference_sdf_file_name,
            core_atom_mapping_dict_list=self.core_atom_mapping_dict_list,
            remove_temp_files=self.remove_temp_files,
            working_dir_name=self.working_dir_name
        )

        unidock_ligand_topology_builder.generate_batch_ligand_topology()
        unidock_ligand_topology_builder.get_summary_ligand_info_dict()

        ## combine inputs into one json file to engine
        system_info_dict = {}
        system_info_dict['score'] = ['vina', 'gaff2']
        system_info_dict.update(unidock_receptor_topology_builder.receptor_info_summary_dict)
        system_info_dict.update(unidock_ligand_topology_builder.total_ligand_info_summary_dict)
        self.unidock2_input_json_file_name = os.path.join(self.working_dir_name, 'system_inputs_unidock2.json')

        with open(self.unidock2_input_json_file_name, 'w') as system_json_file: 
            json.dump(system_info_dict, system_json_file)

        ## run ud2 engine - call the pipeline directly with parameters
        run_docking_pipeline(
            json_file_path=self.unidock2_input_json_file_name,
            output_dir=self.unidock2_output_working_dir_name,
            center_x=float(self.target_center[0]),
            center_y=float(self.target_center[1]),
            center_z=float(self.target_center[2]),
            size_x=self.unidock2_option_dict['Settings']['size_x'],
            size_y=self.unidock2_option_dict['Settings']['size_y'],
            size_z=self.unidock2_option_dict['Settings']['size_z'],
            task=self.unidock2_option_dict['Settings']['task'],
            search_mode=self.unidock2_option_dict['Settings']['search_mode'],
            cluster=self.unidock2_option_dict['Advanced']['cluster'],
            exhaustiveness=self.unidock2_option_dict['Advanced']['exhaustiveness'],
            randomize=self.unidock2_option_dict['Advanced']['randomize'],
            mc_steps=self.unidock2_option_dict['Advanced']['mc_steps'],
            opt_steps=self.unidock2_option_dict['Advanced']['opt_steps'],
            tor_prec=self.unidock2_option_dict['Advanced']['tor_prec'],
            box_prec=self.unidock2_option_dict['Advanced']['box_prec'],
            refine_steps=self.unidock2_option_dict['Advanced']['refine_steps'],
            num_pose=self.unidock2_option_dict['Advanced']['num_pose'],
            rmsd_limit=self.unidock2_option_dict['Advanced']['rmsd_limit'],
            energy_range=self.unidock2_option_dict['Advanced']['energy_range'],
            seed=self.unidock2_option_dict['Advanced']['seed'],
            use_tor_lib=self.unidock2_option_dict['Advanced']['use_tor_lib'],
            constraint_docking=self.template_docking or self.covalent_ligand,
            gpu_device_id=self.unidock2_option_dict['Hardware']['gpu_device_id'],
            max_gpu_memory=self.unidock2_option_dict['Hardware']['max_gpu_memory'],
            ncpu=self.unidock2_option_dict['Hardware']['ncpu']
        )

        ## generate output ud2 pose sdf
        unidock2_pose_json_file_name_raw_list = os.listdir(self.unidock2_output_working_dir_name)
        unidock2_pose_json_file_name_list = [os.path.join(self.unidock2_output_working_dir_name, unidock2_pose_json_file_name_raw) for unidock2_pose_json_file_name_raw in unidock2_pose_json_file_name_raw_list]
        unidock_pose_writer = UnidockLigandPoseWriter(
            unidock_ligand_topology_builder.ligand_mol_list,
            unidock2_pose_json_file_name_list,
            covalent_ligand=self.covalent_ligand,
            working_dir_name=self.working_dir_name
        )

        unidock_pose_writer.generate_docking_pose_sdf()

        self.unidock2_pose_sdf_file_name = unidock_pose_writer.docking_pose_sdf_file_name
