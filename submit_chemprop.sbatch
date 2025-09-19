#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --job-name=chemprop_cmd
##SBATCH --array=0-9
##SBATCH --partition=h100
#SBATCH --partition=l40s
#SBATCH --output=chemprop_output_%A_%a.out
#SBATCH --error=chemprop_error_%A_%a.err

# Load modules or activate environment
module purge
conda activate chemprop
python -c 'import torch; print(torch.cuda.is_available())'

srun python src/chemprop.py --cmd 

