wandb enabled
#wandb disabled
python train.py --device='cuda' \
                --experiment_name='run-gpu' \
                --num_epochs=100 \
                #--subset 4000 \
                #--wandb_name layers-in-prediction-4
