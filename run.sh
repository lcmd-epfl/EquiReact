wandb enabled
wandb disabled
python train.py --device='cuda' \
                --experiment_name='run-gpu2' \
                --subset 100 \
                --num_epochs=2 \
                --dataset cyclo \
                --atom_mapping \
                --noH \
#                --checkpoint logs/run-gpu/231006-135410.579171-xe.best_checkpoint.pt \
                #--rxnmapper \
                #--process \
