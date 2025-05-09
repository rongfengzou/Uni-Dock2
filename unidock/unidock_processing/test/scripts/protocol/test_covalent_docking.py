import os
import shutil
import uuid
import pytest

from unidock_processing.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
)


TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "covalent_docking"
)


@pytest.mark.parametrize(
    "receptor,ligand,covalent_residue_atom_info_list,pocket_center",
    [
        (
            os.path.join(TEST_DATA_DIR, "1EWL", "1EWL_prepared.pdb"),
            [os.path.join(TEST_DATA_DIR, "1EWL", "covalent_mol.sdf")],
            [
                ["", "CYX", 25, "CA"],
                ["", "CYX", 25, "CB"],
                ["", "CYX", 25, "SG"],
            ],
            (8.411, 13.047, 6.811),
        ),
    ]
)
def test_covalent_docking(
    receptor,
    ligand,
    covalent_residue_atom_info_list,
    pocket_center,
):
    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f"./tmp-{uuid.uuid4()}")
    os.mkdir(working_dir_name)

    unidock_protocol_runner = UnidockProtocolRunner(
        receptor,
        ligand,
        target_center=pocket_center,
        box_size=box_size,
        covalent_ligand=True,
        covalent_residue_atom_info_list=covalent_residue_atom_info_list,
        preserve_receptor_hydrogen=True,
        working_dir_name=working_dir_name,
    )

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.unidock2_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)
