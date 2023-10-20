#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=24:00:00
#SBATCH --job-name=chemprop

module purge

conda activate chemprop-patched
python -c 'import torch; print(torch.cuda.is_available())'

../../src/cgr.py -c --rxnmapper --scaffold
