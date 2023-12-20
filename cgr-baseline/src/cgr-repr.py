#!/usr/bin/env python3

import sys
import argparse as ap
import numpy as np
import pandas as pd
import tqdm
import chemprop
from chemprop.features.featurization import set_reaction


parser = ap.ArgumentParser()
parser.add_argument('--column', default='rxn_smiles_mapped', help='csv file column to use, should be the same as used for training')
parser.add_argument('--reaction_mode', default='reac_diff', help='should be the same as used for training, do not change')
parser.add_argument('--checkpoint', default='fold_0/model_0/model.pt', help='path to the checkpoint')
parser.add_argument('--prediction', action='store_true', help='predict targets')
parser.add_argument('--representation', action='store_true', help='save representations')
parser.add_argument('--mpn_path', default='MPN.npy', help='path to save the output of the MPNN portion of the model')
parser.add_argument('--last_ffn_path', default='last_FFN', help='path to save the the input to the final readout layer')
args = parser.parse_args()

arguments = [
    '--test_path', '/dev/null',
    '--preds_path', '/dev/null',
    '--checkpoint_path', args.checkpoint,
]

chemprop_args = chemprop.args.PredictArgs().parse_args(arguments)

set_reaction(True, args.reaction_mode)  # because if first creates the model and then reads its parameters
model_objects = chemprop.train.load_model(args=chemprop_args)
tr_args, pred_args, models, scalers, tasks, names = model_objects

df = pd.read_csv(tr_args.data_path, index_col=0)
smiles = df[args.column].to_numpy()

if args.representation:
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

if args.prediction:
    for smi in tqdm.tqdm(smiles):
        p = chemprop.train.make_predictions(args=chemprop_args, smiles=[[smi]], model_objects=model_objects)
        print(p)
