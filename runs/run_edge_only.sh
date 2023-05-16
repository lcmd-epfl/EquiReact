wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='no-node-score' \
                --num_epochs=500 \
                --wandb_name='no-node-score' \
		--sum_mode='edge' 
