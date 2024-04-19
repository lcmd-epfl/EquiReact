#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00
#SBATCH --job-name=equireact
#SBATCH --array=0-9
#SBATCH --exclude=i39

SEED=$(( $SLURM_ARRAY_TASK_ID + 123 ))

module purge
conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \
--device cuda \
--experiment_name final-cv-80 \
--CV 1 \
--num_epochs 512 \
--seed ${SEED} \
--logdir /scratch/izar/briling/cv \
--dataset proparg \
--train_frac 0.8 \
--combine_mode diff \
--distance_emb_dim 64 \
--dropout_p 0.05 \
--graph_mode vector \
--lr 0.001 \
--max_neighbors 10 \
--n_conv_layers 3 \
--n_s 64 \
--n_v 64 \
--radius 5 \
--sum_mode node \
--weight_decay 0.00001 \
--splitter random \
--noH \
--wandb_name cv10-proparg-inv-random-noH-dft-normal-ns64-nv64-d64-l3-vector-diff-node \
--invariant \

