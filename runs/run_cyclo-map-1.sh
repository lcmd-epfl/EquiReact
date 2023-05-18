wandb enabled
#wandb disabled
srun python train.py --device='cuda' \
                --experiment_name='new-cyclo-mapped' \
                --num_epochs=3000 \
                --n_s=48 \
                --n_v=48 \
                --distance_emb_dim=16 \
                --radius=5.0 \
                --wandb_name=new-cyclo-mapped-ns-48-nv-48-emb16 \
                --atom_mapping=True \
                #--subset 64 \
                #--process True \
