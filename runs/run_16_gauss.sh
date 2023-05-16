wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='16-gauss' \
                --num_epochs=500 \
                --wandb_name='16-gauss' \
		--distance_emb_dim=16
