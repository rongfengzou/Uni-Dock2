# UniDock2
GPU-accelerated molecular docking software: Uni-Dock 2

## Install 
We have prepared an install script for simple installation.
It will create a new conda environment with `Python 3.10`, `CMake 3.31` and `CUDA 11.8`, and compile the C++ engine automatically. Please make sure that there are C++/C compilers on your machine.

And ensure your `.condarc` file contains only these two channels: `defaults` and `conda-forge`.

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
