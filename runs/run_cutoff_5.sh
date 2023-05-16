wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='cutoff-5' \
                --num_epochs=500 \
                --wandb_name='cutoff-5' \
		--sum_mode='node' 
