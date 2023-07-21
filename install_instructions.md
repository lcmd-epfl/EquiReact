# Installation of GPU environment for izar cluster
Assuming a fresh conda environment.

```commandline
module load gcc/8.4.0-cuda cuda/10.2
conda install python=3.10.10
pip install scipy numpy
conda config --add channels pyg
conda config --add channels nvidia
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch -c nvidia
conda install networkx==2.8.4 h5py==3.7
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install e3nn
conda install -c conda-forge rdkit=2023.03.1
pip install pyaml wandb
conda install pyg
```

Note in particular the rdkit version!!
