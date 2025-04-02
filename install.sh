# create user's own conda environment

conda create -n ud2pub python=3.10
conda activate ud2pub

mamba install -y ipython ipykernel ipywidgets requests numba pathos tqdm jinja2 numpy pandas scipy
mamba install -y rdkit openmm mdanalysis openbabel pyyaml networkx ipycytoscape pdbfixer
mamba install -y -c nvidia/label/cuda-11.8.0 cuda # for cuda toolkit
mamba install -y msys_viparr_lpsolve55 ost_promod fftw gemmi==0.6.6 mdtraj ambertools_stable \
	-c http://quetz.dp.tech:8088/get/baymax --no-repodata-use-zst

conda deactivate
conda activate ud2pub

# [Block] For DP Tech Staff
git clone git@git.dp.tech:smallmolecule/fepfixer.git
cd fepfixer
pip install .
cd ..
git clone git@git.dp.tech:smallmolecule/uni-fep.git -b 64-further-refactor-uni-fep
cd uni-fep/unitop
pip install .
cd ../..
# [Block End]


# C++
mamba install -y cmake=3.31
cd unidock/unidock_engine
mkdir build
cd build
cmake ../ud2 -DCMAKE_BUILD_TYPE=Release
make ud2 -j
cd ..
cd ../..

python setup.py install
# conda deactivate then activate
