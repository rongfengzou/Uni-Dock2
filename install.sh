#!/bin/bash

# Three options
read -p "Enter the new conda environment name [ud2pub]: " env_name
env_name=${env_name:-ud2pub}

read -p "Compile C++ components? (y/n) [y]: " compile_cpp
compile_cpp=${compile_cpp:-y}

# Prepare conda environment
mamba create -n $env_name python=3.10 -y
eval "$(mamba shell hook --shell zsh)"
mamba activate $env_name
echo "check python path: $(which python)"

mamba install -y ipython ipykernel ipywidgets requests numba pathos tqdm jinja2 numpy pandas scipy pathos
mamba install -y rdkit openmm mdanalysis openbabel pyyaml networkx ipycytoscape pdbfixer cuda-version=12.1
#mamba install -y -c nvidia/label/cuda-11.8.0 cuda
mamba install -y msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax

echo "check python path: $(which python)"

# C++ Engine Compilation
if [[ "$compile_cpp" == [Yy]* ]]; then
    echo "Compiling C++ engine..."
#    mamba install -y cmake
    cd unidock/unidock_engine
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCONDA_PREFIX=$CONDA_PREFIX
    make ud2 -j
    cd ..
    cd ../..
    python setup.py install
fi

echo "Uni-Dock 2 Installation completed!"
