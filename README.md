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

## Running EquiReact 
Example files for running 10-fold CV runs with 80/10/10 splits for either random or scaffold splits are provided in `submit-cv/`. In essence, `train.py` is run with the optimized hyperparameters, split arguments, and informaton on where to save models and results. 
There is an argument in `train.py` to also run evaluation on the test set after training (`eval_on_test_split`) but to run evaluation after training, specifying a saved model, one can use `evaluate.py`

To optimize hyperparameters, a sweep can be run with `wandb` using `sweep.py`.

Note that these files currently run on the three datasets studied in the paper (Cyclo-23-TS, GDB7-22-TS and Proparg-21-TS) with corresponding dataloaders in `process/dataloader_<dataset>.py`. To run on a different dataset, a dataloader needs to be written and the train code slightly modified to handle the new set.


## Analyzing representations
If desired, the learned representation can be extracted using `representation.py`, which may be interesting for model interpretation or other downstream applications.

## Baselines
To run the baselines CGR and SLATM_d, the former on GPU and the latter on CPU, two additional installation files are provided: <CGR> and `requirements_fingerprints.txt` (assuming these will be run in separate environments) 
