#!/bin/bash

# Three options
read -p "Enter the new conda environment name [ud2pub]: " env_name
env_name=${env_name:-ud2pub}

read -p "Are you a DP Tech staff (access to private tools)? (y/n) [n]: " is_dp_staff
is_dp_staff=${is_dp_staff:-n}

read -p "Compile C++ components? (y/n) [y]: " compile_cpp
compile_cpp=${compile_cpp:-y}

# Prepare conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda create -n $env_name python=3.10 -y
conda activate $env_name
echo "check python path: $(which python)"

mamba install -y ipython ipykernel ipywidgets requests numba pathos tqdm jinja2 numpy pandas scipy
mamba install -y rdkit openmm mdanalysis openbabel pyyaml networkx ipycytoscape pdbfixer
mamba install -y -c nvidia/label/cuda-11.8.0 cuda
mamba install -y msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax

conda activate $env_name
echo "check python path: $(which python)"

# DP Tech staff only
if [[ "$is_dp_staff" == [Yy]* ]]; then
    echo "Installing DP internal components..."
    git clone git@git.dp.tech:smallmolecule/fepfixer.git
    cd fepfixer
    pip install .
    cd ..
    git clone git@git.dp.tech:smallmolecule/uni-fep.git -b 64-further-refactor-uni-fep
    cd uni-fep/unitop
    pip install .
    cd ../..
fi

# C++ Engine Compilation
if [[ "$compile_cpp" == [Yy]* ]]; then
    echo "Compiling C++ engine..."
    mamba install -y cmake=3.31
    cd unidock/unidock_engine
    mkdir build
    cd build
    cmake ../ud2 -DCMAKE_BUILD_TYPE=Release
    make ud2 -j
    cd ..
    cd ../..
    python setup.py install
fi

echo "Uni-Dock 2 Installation completed!"



