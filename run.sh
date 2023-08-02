wandb enabled
#wandb disabled
python train_mapper.py --device='cuda' \
                --experiment_name='test-mapper' \
                --num_epochs=100 \
                --CV=1 \
                --wandb_name='test-mapper'
