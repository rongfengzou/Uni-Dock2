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


# [Security Check]
expected_python="$CONDA_BASE/envs/$env_name/bin/python"
current_python=$(which python)
if [[ "$current_python" != "$expected_python" ]]; then
    echo -e "\033[31mERROR: Failed to activate the new conda environment!\033[0m"
    echo "Current Python: $current_python"
    echo "Expected Python: $expected_python"
    exit 1
else
    echo -e "Environment verified."
fi

conda install -y ipython ipykernel ipywidgets requests numba pathos tqdm jinja2 numpy pandas scipy pathos
conda install -y rdkit openmm mdanalysis openbabel pyyaml networkx ipycytoscape pdbfixer cuda-version=12.1
conda install -y msys_viparr_lpsolve55 ambertools_stable -c http://quetz.dp.tech:8088/get/baymax --no-repodata-use-zst

echo "check python path: $(which python)"

# C++ Engine Compilation
if [[ "$compile_cpp" == [Yy]* ]]; then
    echo "Compiling C++ engine..."
    conda install -y cmake=3.27
    conda install nvidia/label/cuda-12.1.1::cuda-nvcc nvidia/label/cuda-12.1.1::cuda-cudart-static nvidia/label/cuda-12.1.1::libcurand nvidia/label/cuda-12.1.1::libcurand-static
    cd unidock/unidock_engine
    mkdir build
    cd build
    #export PATH=$CONDA_PREFIX/bin:$PATH # if your conda path is overridden by system path
    cmake .. -DCMAKE_BUILD_TYPE=Release # -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc
    make ud2 -j
    cd ../../..
    python setup.py install
fi

echo "Uni-Dock 2 Installation completed!"
