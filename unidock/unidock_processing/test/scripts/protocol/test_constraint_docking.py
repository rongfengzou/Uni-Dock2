import os
import shutil
import uuid
import pytest
import yaml

from unidock.unidock_processing.unidocktools.unidock_protocol_runner import UnidockProtocolRunner

@pytest.fixture
def template_configurations():
    return os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'unidock_configurations.yaml')

@pytest.fixture
def receptor_constraint_docking():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'Bace.pdb')

@pytest.fixture
def ligand_constraint_docking():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking')
    data_file_name_list = os.listdir(data_file_dir_name)

    ligand_sdf_file_name_list = []
    for data_file_name in data_file_name_list:
        if data_file_name.startswith('CAT') and data_file_name.endswith('sdf'):
            ligand_sdf_file_name = os.path.join(data_file_dir_name, data_file_name)
            ligand_sdf_file_name_list.append(ligand_sdf_file_name)

    return ligand_sdf_file_name_list

@pytest.fixture
def reference_constraint_docking():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'reference.sdf')

@pytest.fixture
def pocket_center_constraint_docking():
    return (14.786, -0.626, -1.088)

def test_constraint_docking(receptor_constraint_docking,
                            ligand_constraint_docking,
                            reference_constraint_docking,
                            pocket_center_constraint_docking,
                            template_configurations):

    with open(template_configurations, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock2_option_dict['Settings']['size_x'] = box_size[0]
    unidock2_option_dict['Settings']['size_y'] = box_size[1]
    unidock2_option_dict['Settings']['size_z'] = box_size[2]
    unidock2_option_dict['Preprocessing']['template_docking'] = True
    unidock2_option_dict['Preprocessing']['reference_sdf_file_name'] = reference_constraint_docking
    unidock2_option_dict['Preprocessing']['core_atom_mapping_dict_list'] = None
    unidock2_option_dict['Preprocessing']['working_dir_name'] = working_dir_name

    test_configuration_file_name = os.path.join(working_dir_name, 'unidock_configurations.yaml')
    with open(test_configuration_file_name, 'w') as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_protocol_runner = UnidockProtocolRunner(receptor_constraint_docking,
                                                    ligand_constraint_docking,
                                                    target_center=pocket_center_constraint_docking,
                                                    option_yaml_file_name=test_configuration_file_name)

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    shutil.rmtree(working_dir_name, ignore_errors=True)
