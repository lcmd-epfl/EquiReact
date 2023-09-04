#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00
#SBATCH --job-name=equireact
#SBATCH --array=0-9

SEED=$(( $SLURM_ARRAY_TASK_ID + 123 ))

module purge
module load gcc/8.4.0-cuda cuda/10.2

conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'

wandb enabled
#wandb disabled
python train.py --device cuda              \
                --experiment_name final-cv \
                --CV 1                     \
                --num_epochs 512           \
                --seed ${SEED}             \
                --dataset "proparg"        \
                --combine_mode "diff"      \
                --distance_emb_dim 16      \
                --dropout_p 0.1            \
                --graph_mode "vector"      \
                --max_neighbors 50         \
                --n_conv_layers 3          \
                --n_s 64                   \
                --n_v 16                   \
                --radius 5                 \
                --sum_mode "node"          \
                --wandb_name=cv10-proparg-noH-ns64-nv16-d16-layers3-vector-diff-node-crossattention \
                --attention cross \
                --noH \
                --logdir /scratch/izar/briling/cv  \
