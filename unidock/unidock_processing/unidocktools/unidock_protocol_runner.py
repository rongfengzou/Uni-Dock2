from typing import List, Tuple, Dict, Optional, Any
import os
import json
from shutil import rmtree

from rdkit import Chem

from unidock_engine.api.python.pipeline import run_docking_pipeline
from unidock_processing.unidocktools.unidock_receptor_topology_builder import UnidockReceptorTopologyBuilder
from unidock_processing.unidocktools.unidock_ligand_topology_builder import UnidockLigandTopologyBuilder
from unidock_processing.unidocktools.unidock_ligand_pose_writer import UnidockLigandPoseWriter
from unidock_processing.ligand_topology import utils


class UnidockProtocolRunner(object):
    def __init__(self,
                 receptor_file_name: str,
                 ligand_sdf_file_name_list: List[str],
                 target_center: Tuple[float, float, float],
                 template_docking: bool = False,
                 reference_sdf_file_name: Optional[str] = None,
                 covalent_ligand: bool = False,
                 covalent_residue_atom_info_list: Optional[List[Dict[str, Any]]] = None,
                 preserve_receptor_hydrogen: bool = False,
                 remove_temp_files: bool = True,
                 working_dir_name: str = 'unidock_workdir',
                 core_atom_mapping_dict_list: Optional[List[Optional[Dict[int, int]]]] = None,
                 size_x: float = 25.0,
                 size_y: float = 25.0,
                 size_z: float = 25.0,
                 task: str = 'screen',
                 search_mode: str = 'free',
                 exhaustiveness: int = 512,
                 randomize: bool = True,
                 mc_steps: int = 200,
                 opt_steps: int = 5,
                 tor_prec: float = 0.1,
                 box_prec: float = 100.0,
                 refine_steps: int = 0,
                 num_pose: int = 10,
                 rmsd_limit: float = 1.0,
        ) -> None:

        self.receptor_file_name: str = receptor_file_name
        self.ligand_sdf_file_name_list: List[str] = ligand_sdf_file_name_list
        self.target_center: Tuple[float, float, float] = target_center
        
        # Configuration parameters
        self.template_docking: bool = template_docking
        self.reference_sdf_file_name: Optional[str] = reference_sdf_file_name
        self.covalent_ligand: bool = covalent_ligand
        self.covalent_residue_atom_info_list: Optional[List[Dict[str, Any]]] = covalent_residue_atom_info_list
        self.preserve_receptor_hydrogen: bool = preserve_receptor_hydrogen
        self.remove_temp_files: bool = remove_temp_files
        
        # Docking parameters
        self.size_x: float = size_x
        self.size_y: float = size_y
        self.size_z: float = size_z
        self.task: str = task
        self.search_mode: str = search_mode
        self.exhaustiveness: int = exhaustiveness
        self.randomize: bool = randomize
        self.mc_steps: int = mc_steps
        self.opt_steps: int = opt_steps
        self.tor_prec: float = tor_prec
        self.box_prec: float = box_prec
        self.refine_steps: int = refine_steps
        self.num_pose: int = num_pose
        self.rmsd_limit: float = rmsd_limit

        self.working_dir_name: str = os.path.abspath(working_dir_name)
        self.unidock2_output_working_dir_name: str = os.path.join(self.working_dir_name, 'unidock2_output')
        self.unidock2_input_json_file_name: str = ""
        self.unidock2_pose_sdf_file_name: str = ""

        if os.path.isdir(self.unidock2_output_working_dir_name):
            rmtree(self.unidock2_output_working_dir_name, ignore_errors=True)
            os.mkdir(self.unidock2_output_working_dir_name)
        else:
            os.mkdir(self.unidock2_output_working_dir_name)

        # Process core atom mapping dict list
        if core_atom_mapping_dict_list is None:
            self.core_atom_mapping_dict_list: Optional[List[Optional[Dict[int, int]]]] = None
        else:
            num_molecules: int = len(core_atom_mapping_dict_list)
            self.core_atom_mapping_dict_list = [None] * num_molecules

            for mol_idx in range(num_molecules):
                raw_core_atom_mapping_dict = core_atom_mapping_dict_list[mol_idx]
                if raw_core_atom_mapping_dict is None:
                    self.core_atom_mapping_dict_list[mol_idx] = None
                else:
                    core_atom_mapping_dict = {int(reference_atom_idx): int(query_atom_idx) for reference_atom_idx, query_atom_idx in raw_core_atom_mapping_dict.items()}
                    self.core_atom_mapping_dict_list[mol_idx] = core_atom_mapping_dict

        if self.template_docking and self.target_center == (0.0, 0.0, 0.0):
            reference_mol = Chem.SDMolSupplier(self.reference_sdf_file_name, removeHs=True)[0]
            self.target_center = tuple(utils.calculate_center_of_mass(reference_mol))

    def run_unidock_protocol(self) -> str:
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
        system_info_dict: Dict[str, Any] = {}
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
            center_x=self.target_center[0],
            center_y=self.target_center[1],
            center_z=self.target_center[2],
            size_x=self.size_x,
            size_y=self.size_y,
            size_z=self.size_z,
            task=self.task,
            search_mode=self.search_mode,
            exhaustiveness=self.exhaustiveness,
            randomize=self.randomize,
            mc_steps=self.mc_steps,
            opt_steps=self.opt_steps,
            tor_prec=self.tor_prec,
            box_prec=self.box_prec,
            refine_steps=self.refine_steps,
            num_pose=self.num_pose,
            rmsd_limit=self.rmsd_limit,
            constraint_docking=self.template_docking or self.covalent_ligand,
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
        return self.unidock2_pose_sdf_file_name
