wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='ns-32' \
                --num_epochs=500 \
                --wandb_name='ns-32' \
		--n_s=32
