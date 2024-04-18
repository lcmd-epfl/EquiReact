#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
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
--dataset cyclo \
--train_frac 0.8 \
--combine_mode diff \
--distance_emb_dim 48 \
--dropout_p 0.1 \
--graph_mode energy \
--lr 0.001 \
--max_neighbors 50 \
--n_conv_layers 2 \
--n_s 64 \
--n_v 48 \
--radius 2.5 \
--sum_mode node \
--weight_decay 0.00001 \
--two_layers_atom_diff \
--atom_mapping \
--splitter sizedesc \
--noH \
--wandb_name cv10-cyclo-sizedesc-noH-dft-true-ns64-nv48-d48-l2-energy-diff-node \

