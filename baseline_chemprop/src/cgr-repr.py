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
parser.add_argument('--checkpoint', default='fold_0/fold_0/model_0/model.pt', help='path to the checkpoint')
parser.add_argument('--fingerprint_type', default='MPN', help='MPN / last_FFN')
parser.add_argument('--data_path', default='../../csv/gdb.csv', help='full dataset csv path')
args = parser.parse_args()

arguments = [
    '--test_path', '/dev/null',
    '--preds_path', '/dev/null',
    '--checkpoint_path', args.checkpoint,
    '--fingerprint_type', args.fingerprint_type,
]

chemprop_args = chemprop.args.FingerprintArgs().parse_args(arguments)

df = pd.read_csv(args.data_path, index_col=0)
smiles = df[args.column].to_numpy()

#x = []
#for smi in tqdm.tqdm(smiles):
#    print()
#    x.append(chemprop.train.molecule_fingerprint.molecule_fingerprint(args=chemprop_args, smiles=[[smi]]).squeeze())
#    print()
#x = np.vstack(x)


smiles = [[smi] for smi in smiles]
x = chemprop.train.molecule_fingerprint.molecule_fingerprint(args=chemprop_args, smiles=smiles).squeeze().astype(np.float32)


print(x.shape)
np.save(f'{args.fingerprint_type}.npy', x)
