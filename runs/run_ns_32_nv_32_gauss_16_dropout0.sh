wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='ns-32-nv-32-16gauss-dropout0' \
                --num_epochs=2000 \
                --wandb_name='ns-32-nv-32-16gauss-dropout0' \
		--n_s=32 \
		--n_v=32 \
		--distance_emb_dim=16 \
		--radius=5. \
		--dropout_p=0.
