wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='ns-48' \
                --num_epochs=1000 \
                --wandb_name='ns-48-16-gauss' \
		--n_s=48 \
		--distance_emb_dim=16 \
		--radius=5.
