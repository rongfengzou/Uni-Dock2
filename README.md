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
* command line example:
```
unidock2 -r receptor.pdb -lb ligand_batch.dat --center 5.12 18.35 37.36 --configurations test.yaml

unidock2 -v
unidock2 -h
```

To use Uni-Dock 2 for python scripting, follow this example:
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
    yaml_file = os.path.abspath(sys.argv[3]) # input configuration

    target_center = (5.12, 18.35, 37.36)
    box_size = (30.0, 30.0, 30.0)

    unidock_protocol_runner = UnidockProtocolRunner(fp_pdb,
                                                    [fp_sdf], # One sdf containing many ligands is allowed; Many sdf files are also allowed.
                                                    target_center=target_center,
                                                    option_yaml_file_name=yaml_file)

    unidock_protocol_runner.run_unidock_protocol()
```

After execution, the output SDF file will contain all poses, scores, and other relevant data.
### Constraint docking
```python
    dp_data = 'align_input'
    fp_sdf = [os.path.join(dp_data, "new.sdf")]
    yaml_file = os.path.abspath("constraint.yaml") # input configuration

    target_center = (18.974, 20.621, 15.106)
    box_size = (30.0, 30.0, 30.0)

    unidock_protocol_runner = UnidockProtocolRunner(fp_pdb,
                                                    fp_sdf,
                                                    target_center=target_center,
                                                    option_yaml_file_name=yaml_file)

    unidock_protocol_runner.run_unidock_protocol()
```
