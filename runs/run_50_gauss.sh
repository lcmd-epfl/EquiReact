wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='50-gauss' \
                --num_epochs=500 \
                --wandb_name='50-gauss' \
		--distance_emb_dim=50
