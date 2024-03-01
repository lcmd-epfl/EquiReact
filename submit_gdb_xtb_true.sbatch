#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=72:00:00
#SBATCH --job-name=gdb-xtb
#SBATCH --array=0-9
#SBATCH --exclude=i39

SEED=$(( $SLURM_ARRAY_TASK_ID + 123 ))

module purge

conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'

wandb disabled
python train.py --device cuda              \
                --experiment_name benchmark-gdb-xtb \
                --CV 1                     \
                --num_epochs 512           \
                --seed ${SEED}             \
                --dataset "gdb"            \
                --combine_mode "diff"      \
                --distance_emb_dim 48      \
                --dropout_p 0.05           \
                --graph_mode "vector"      \
                --max_neighbors 50         \
                --n_conv_layers 3          \
                --n_s 64                   \
                --n_v 64                   \
                --radius 2.5               \
                --sum_mode "node"          \
                --atom_mapping \
                --two_layers_atom_diff \
                --logdir /scratch/izar/vangerwe/cv  \
                --lr 1e-3 --weight_decay 1e-5 \
                --xtb \
		--noH \
		--train_frac 0.8 \
		--eval_on_test_split \
