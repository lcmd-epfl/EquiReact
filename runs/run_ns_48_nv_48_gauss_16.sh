wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='ns-48-nv-48-16gauss' \
                --num_epochs=2000 \
                --wandb_name='ns-48-nv-48-16gauss' \
		--n_s=48 \
		--n_v=48 \
		--distance_emb_dim=16 \
		--radius=5.
