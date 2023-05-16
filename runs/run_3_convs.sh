wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='3-conv' \
                --num_epochs=500 \
                --wandb_name='3-conv' \
		--n_conv_layers=3
