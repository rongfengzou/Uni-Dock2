# UniDock2
GPU-accelerated molecular docking software: Uni-Dock 2


## Install 
We have prepared an install script for simple installation.
It will create a new conda environment with `Python 3.10`, `CMake 3.31` and `CUDA 11.8`, and compile the C++ engine automatically. Please make sure that there are C++/C compilers on your machine.

You can modify the environment name in the script as you like. Then perform the script:
```sh
chmod +x ./install.sh
./install.sh
```


## Usage
To use Uni-Dock 2, follow this example:
* Ensure you have the required input files: a PDB file and an SDF file.
* Define the target center and box size for docking.
* Call the UnidockProtocolRunner with the appropriate parameters.

```python
from unidock.unidock_processing.unidocktools.unidock_protocol_runner import UnidockProtocolRunner
import sys
import os

if __name__ == "__main__":
    assert(len(sys.argv) > 3)
    fp_pdb  = os.path.abspath(sys.argv[1])
    fp_sdf  = os.path.abspath(sys.argv[2]) 
    dp_res  = os.path.abspath(sys.argv[3]) # output directory

    target_center = (5.12,18.35,37.36)
    box_size = (30.0, 30.0, 30.0)


    unidock_protocol_runner = UnidockProtocolRunner(fp_pdb,
                                                    [fp_sdf], # One sdf containing many ligands is allowed; Many sdf files are also allowed.
                                                    target_center=target_center,
                                                    box_size=box_size,
                                                    template_docking=False, # constraint docking
                                                    reference_sdf_file_name=None,
                                                    core_atom_mapping_dict_list=None,
                                                    covalent_ligand=False,
                                                    covalent_residue_atom_info_list=None,
                                                    working_dir_name=dp_res)
    unidock_protocol_runner.run_unidock_protocol()
```

After execution, the output SDF file will contain all poses, scores, and other relevant data.
### Constraint docking
```python
    dp_data = 'align_input'
    fp_ref = os.path.join(dp_data, "reference.sdf")
    fp_sdf = [os.path.join(dp_data, "new.sdf")]
    fp_pdb = os.path.join(dp_data, 'protein.pdb')
    dp_res = "result"

    target_center = (18.974199771881104, 20.620699882507324, 15.10605001449585)
    box_size = (22.5, 22.5, 22.5)

    unidock_protocol_runner = UnidockProtocolRunner(fp_pdb,
                                                    fp_sdf,
                                                    target_center=target_center,
                                                    box_size=box_size,
                                                    template_docking=True, # constraint docking
                                                    reference_sdf_file_name=fp_ref,
                                                    core_atom_mapping_dict_list=None,
                                                    covalent_ligand=False,
                                                    covalent_residue_atom_info_list=None,
                                                    working_dir_name=dp_res,
                                                    remove_temp_files=True
                                                    )
    unidock_protocol_runner.run_unidock_protocol()
```

