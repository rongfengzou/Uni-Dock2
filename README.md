# UniDock2
GPU-accelerated molecular docking software: Uni-Dock 2

---
## Installation
### Easy installation via install.sh
We provide an install script to simplify the setup process. This script will

1. Create a **new conda environment** with `Python 3.10`, `CMake 3.31` and `CUDA 12.1`, along with all other necessary python packages
2. Automatically compile the C++ engine
3. Install `unidock2` command to this environment.

#### Important Notes
* Run this script outside of your base environment to prevent accidental configuration conflict.
* Ensure your `.condarc` file contains only these two channels: `defaults` and `conda-forge`.

#### Usage
```sh
chmod +x ./install.sh
./install.sh
```
After successful installation, the `unidock2` command will be available in your new environment.


### Manual Build Instruction
#### 1. Clone Repository
```sh
git clone https://github.com/dptech-corp/Uni-Dock.git
```
#### 2. Compile Engine
##### Prerequisites
* `CUDA toolkit >= 11.8`
* `CMake >= 3.27`
* C++ compiler

##### Build
```sh
cd unidock/unidock_engine
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make ud2 -j
```
Verify the path `unidock/unidock_engine/build/bin/ud2`,  which should match that in `setup.py` 

#### 3. Install python package
##### Dependencies
`
ipython ipykernel ipywidgets requests numba pathos tqdm jinja2 numpy pandas scipy pathos rdkit openmm mdanalysis openbabel pyyaml networkx ipycytoscape pdbfixer
`

Additionally, install two packages from our private conda channel:
```sh
conda install msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax --no-repodata-use-zst
```

##### Install
```sh
python setup.py install
```


---
## Usage
Command line example:
```
for running docking:
unidock2 -r receptor.pdb -lb ligand_batch.dat -c 5.12 18.35 37.36 -cf unidock_configurations.yaml

for checking current version:
unidock2 -v

for checking help information:
unidock2 -h
```

The template YAML file path is `unidock/unidock_configurations.yaml`

For example free docking case, please see `unidock/unidock_processing/test/data/free_docking/molecular_docking`
The default configurations should be fine in the example `unidock_configurations.yaml` file, just enter the test case folder and run:
```
unidock2 -r 1G9V_protein_water_cleaned.pdb -l ligand_prepared.sdf -c 5.122 18.327 37.332 -cf unidock_configurations.yaml
```

For example constraint docking case, please see `unidock/unidock_processing/test/data/constraint_docking/automatic_atom_mapping` for automatic atom mapping case and `unidock/unidock_processing/test/data/constraint_docking/manual_atom_mapping` for manual specified mapping case.

Please find the example changes and options in the `unidock_configurations.yaml` file. `template_docking` option should be turned on, `reference_sdf_file_name` should be specified and `core_atom_mapping_dict_list` should be `null` for automatic case, or a list of dict for manual specifying mapping case.

example run:
```
for automatic case:
unidock2 -r Bace.pdb -lb ligand_batch.dat -c 14.786 -0.626 -1.088 -cf unidock_configurations.yaml

for manual case:
unidock2 -r protein.pdb -l ligand.sdf -c 9.028 0.804 21.789 -cf unidock_configurations.yaml
```

For example covalent docking case, please find `unidock/unidock_processing/test/data/covalent_docking/1EWL`
Please find the example changes and options in the `unidock_configurations.yaml` file. `covalent_ligand` option should be turned on, `covalent_residue_atom_info_list` should be specified to tell the algorithm about the covalent warhead residue information. The warhead information is a list of 3 lists. Each one records one residue information of the corresponding heavy atoms. The heavy atoms starts from the warhead atom, the covalent bond starting atom, and the ending atom on the covalent proteihn residue.

example run:
```
for automatic case:
unidock2 -r Bace.pdb -lb ligand_batch.dat -c 14.786 -0.626 -1.088 -cf unidock_configurations.yaml

for manual case:
unidock2 -r 1EWL_prepared.pdb -l covalent_mol.sdf -c 8.411 13.047 6.811 -cf unidock_configurations.yaml
```
