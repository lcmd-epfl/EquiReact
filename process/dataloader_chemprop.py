from chemprop.data.utils import get_data_from_smiles
import pandas as pd
from process.scaffold import scaffold_split
import numpy as np
from rdkit import Chem
import re

def remove_atom_map_number_manual(smiles):
    smi = re.sub(':[0-9]+', '', smiles)
    return smi

def get_scaffold_splits_gdb(data='data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv',
                            shuffle_indices=None,
                            reactant_col='rsmi'):

    df = pd.read_csv(data, index_col=0)
    if shuffle_indices is None:
        shuffle_indices = np.arange(len(df))
    else:
        assert len(shuffle_indices) == len(df), "lost data in shuffle"
    rsmiles = df[reactant_col].to_numpy()
    rsmiles = np.array([remove_atom_map_number_manual(smiles) for smiles in rsmiles])
    rsmiles_shuffled = rsmiles[shuffle_indices]

    dataset = get_data_from_smiles([[x] for x in rsmiles_shuffled])
    train_idx, test_idx, val_idx = scaffold_split(dataset)
    return train_idx, test_idx, val_idx