#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=72:00:00
#SBATCH --job-name=cyclo-xtb
#SBATCH --array=0-9

SEED=$(( $SLURM_ARRAY_TASK_ID + 123 ))

module purge

conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'

wandb disabled
python train.py --device cuda              \
                --experiment_name benchmark-cyclo-xtb \
                --CV 1                     \
                --num_epochs 512           \
                --seed ${SEED}             \
                --dataset "cyclo"          \
                --combine_mode "mlp"       \
                --distance_emb_dim 48      \
                --dropout_p 0.05           \
                --graph_mode "energy"      \
                --max_neighbors 10         \
                --n_conv_layers 3          \
                --n_s 64                   \
                --n_v 48                   \
                --radius 2.5               \
                --sum_mode "both"          \
                --logdir /scratch/izar/vangerwe/cv  \
                --lr 5e-4 --weight_decay 1e-4 \
                --xtb \
		--noH \
		--train_frac 0.8 \
		--eval_on_test_split \
