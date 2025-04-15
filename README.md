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
unidock2 -r receptor.pdb -lb ligand_batch.dat --center 5.12 18.35 37.36 --configurations test.yaml

unidock2 -v
unidock2 -h
```

The template YAML file path is `unidock/unidock_configurations.yaml`
