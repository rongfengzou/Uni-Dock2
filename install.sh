#!/bin/bash

# Three options
read -p "Enter the new conda environment name [ud2pub]: " env_name
env_name=${env_name:-ud2pub}

read -p "Compile C++ components? (y/n) [y]: " compile_cpp
compile_cpp=${compile_cpp:-y}

# Prepare conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda create -n $env_name python=3.10 -y
conda activate $env_name
echo "check python path: $(which python)"

conda install -y mamba==1.5.4 -c conda-forge
mamba install -y ipython ipykernel ipywidgets requests numba pathos tqdm jinja2 numpy pandas scipy
mamba install -y rdkit openmm mdanalysis openbabel pyyaml networkx ipycytoscape pdbfixer
mamba install -y -c nvidia/label/cuda-11.8.0 cuda
mamba install -y msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax # --no-repodata-use-zst

conda activate $env_name
echo "check python path: $(which python)"

# C++ Engine Compilation
if [[ "$compile_cpp" == [Yy]* ]]; then
    echo "Compiling C++ engine..."
    mamba install -y cmake=3.31
    cd unidock/unidock_engine
    mkdir build
    cd build
    cmake ../ud2 -DCMAKE_BUILD_TYPE=Release -DCONDA_PREFIX=$CONDA_PREFIX
    make ud2 -j
    cd ..
    cd ../..
    python setup.py install
fi

echo "Uni-Dock 2 Installation completed!"
