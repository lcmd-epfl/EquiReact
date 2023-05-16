wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='both-score' \
                --num_epochs=500 \
                --wandb_name='both-score' \
		--sum_mode='both' 
