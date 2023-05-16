wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='no-edge-score' \
                --num_epochs=500 \
                --wandb_name='no-edge-score' \
		--sum_mode='node' 
