# EquiReact 

This repo provides the code for the EquiReact model, as well as the raw and pre-processed data from the three datasets (GDB7-22-TS, Cyclo-23-TS and Proparg-21-TS).

## Installation 
For a direct copy of the environment used to run the results: 
`conda create -n <env_name> --file requirements.txt`
But this will depend on the version of CUDA you have available.

Otherwise the key packages to install are as follows, assuming running on a cluster where modules need to be loaded:
```
module load gcc/<version>-cuda cuda/<version>
conda install python=3.10.10
pip install scipy numpy
conda config --add channels pyg
conda config --add channels nvidia
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=<version> -c pytorch -c nvidia
conda install networkx==2.8.4 h5py==3.7
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu<version>.html
pip install e3nn
conda install -c conda-forge rdkit=2023.03.1
pip install pyaml wandb
conda install pyg
```
