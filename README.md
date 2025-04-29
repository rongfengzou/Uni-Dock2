# UniDock2
GPU-accelerated molecular docking software: Uni-Dock 2

---

## Installation
### Method 1: Conda Installation
The easiest way to install UniDock2 is via conda:

#### Prerequisites
* Python = 3.10
* CUDA >= 12.0

```sh
#You can modify the cuda-version to fit your environment.
conda install unidock2 cuda-version=12.0 -c http://quetz.dp.tech:8088/get/baymax -c conda-forge --no-repodata-use-zst 
```

### Method 2: Manual Build
```sh
git clone https://github.com/dptech-corp/Uni-Dock.git
```

#### 1. Build unidock_engine
##### Prerequisites
* CUDA toolkit >= 12.0 (Including nvcc)
* CMake >= 3.27
* C++ compiler
* Pybind11

```sh
cd unidock/unidock_engine
pip install .
cd ../..
```

#### 2. Build unidock_processing
```sh
conda install pyyaml pathos numpy pandas scipy networkx rdkit mdanalysis openbabel pdbfixer openmm cuda-version=12.0 msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax -c conda-forge --no-repodata-use-zst

pip install .
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
