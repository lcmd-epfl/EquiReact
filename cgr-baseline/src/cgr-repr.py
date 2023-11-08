#!/usr/bin/env python3

import sys
import argparse as ap
import numpy as np
import pandas as pd
import tqdm
import chemprop
from chemprop.features.featurization import set_reaction

col = 'rxn_smiles_mapped'  # change if change columns

arguments = [
    '--test_path', '/dev/null',
    '--preds_path', '/dev/null',
    '--checkpoint_path', 'fold_00/model_0/model.pt'
]

args = chemprop.args.PredictArgs().parse_args(arguments)

set_reaction(True, 'reac_diff')  # because if first creates the model and then reads its parameters
model_objects = chemprop.train.load_model(args=args)
tr_args, pred_args, models, scalers, tasks, names = model_objects

print(tr_args)
df = pd.read_csv(tr_args.data_path, index_col=0)
smiles = df[col].to_numpy()

if True:
    x = []
    y = []
    tr_args, pred_args, models, scalers, tasks, names = model_objects
    model = models[0]
    for smi in tqdm.tqdm(smiles):
        x.append(model.fingerprint([[smi]], fingerprint_type='MPN').detach().numpy().squeeze())
        y.append(model.fingerprint([[smi]], fingerprint_type='last_FFN').detach().numpy().squeeze())
    x = np.vstack(x)
    y = np.vstack(y)
    print(x.shape, y.shape)
    np.save('MPN.npy', x)
    np.save('last_FFN.npy', y)

if True:
    for smi in tqdm.tqdm(smiles):
        z = chemprop.train.make_predictions(args=args, smiles=[[smi]], model_objects=model_objects)
        print(z)
