# Installation of GPU environment for izar cluster 
Assuming a fresh conda environment.

```commandline
module load gcc/8.4.0-cuda cuda/10.2
pip install scipy numpy
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 pytorch-cuda=10.2 -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
pip install e3nn
conda install rdkit -c conda-forge
pip install pyaml wandb
```

Hope that works, let me know...