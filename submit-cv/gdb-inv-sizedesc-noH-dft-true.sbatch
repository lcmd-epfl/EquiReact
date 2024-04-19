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
--dataset gdb \
--train_frac 0.8 \
--combine_mode diff \
--distance_emb_dim 32 \
--dropout_p 0.05 \
--graph_mode vector \
--lr 0.0005 \
--max_neighbors 10 \
--n_conv_layers 2 \
--n_s 64 \
--n_v 64 \
--radius 2.5 \
--sum_mode both \
--weight_decay 0.00001 \
--two_layers_atom_diff \
--atom_mapping \
--splitter sizedesc \
--noH \
--wandb_name cv10-gdb-inv-sizedesc-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both \
--invariant \

