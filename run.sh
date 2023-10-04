wandb enabled
wandb disabled
python train.py --device='cuda' \
                --experiment_name='run-gpu' \
                --subset 100 \
                --num_epochs=3 \
                --process \
                --dataset cyclo \
                --atom_mapping \
                #--noH \
                #--rxnmapper \
