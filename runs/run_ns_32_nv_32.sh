wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='ns-32-nv-32-16gauss' \
                --num_epochs=1000 \
                --wandb_name='ns-32-nv-32-16gauss' \
		--n_s=32 \
		--n_v=32 \
		--distance_emb_dim=16 \
		--radius=5.
