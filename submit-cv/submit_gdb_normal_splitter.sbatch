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
#wandb disabled
python train.py --device cuda              \
                --experiment_name final-cv \
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
                --noH                      \
		--splitter='scaffold'      \
                --wandb_name=cv10-LP-gdb-scaffold-ns64-nv64-d48-layers3-vector-diff-node-noH \
                --logdir /scratch/izar/vangerwe/cv  \
                --lr 1e-3 --weight_decay 1e-5 \
