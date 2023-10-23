from chemprop.data.utils import get_data_from_smiles
import pandas as pd
from process.scaffold import scaffold_split
import numpy as np
import re
import warnings

def remove_atom_map_number_manual(smiles):
    smi = re.sub(':[0-9]+', '', smiles)
    return smi

def get_reactant_from_reaction_smi(reaction_smiles):
    return reaction_smiles.split('>>')[0]

def get_scaffold_splits(dataset, shuffle_indices=None, sizes=(0.8, 0.1, 0.1)):
    if dataset == 'gdb':
        data = 'data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv'
    elif dataset == 'cyclo':
        data = 'data/cyclo/cyclo.csv'
    elif dataset == 'proparg':
        data = 'data/proparg/proparg.csv'
    else:
        raise ValueError("dataset has to be a string gdb, cyclo or proparg")
    df = pd.read_csv(data, index_col=0)
    if dataset == 'gdb':
        rsmiles = df['rsmi'].to_numpy()
    elif dataset == 'cyclo' or dataset == 'proparg':
        rsmiles = df['rxn_smiles'].apply(get_reactant_from_reaction_smi).to_numpy()
    if shuffle_indices is None:
        shuffle_indices = np.arange(len(df))
    else:
        bool = len(shuffle_indices) == len(df)
        if not bool:
            warnings.warn("Shuffle indices are not the same len as the original df")

    rsmiles = np.array([remove_atom_map_number_manual(smiles) for smiles in rsmiles])
    rsmiles_shuffled = rsmiles[shuffle_indices]

    dataset = get_data_from_smiles([[x] for x in rsmiles_shuffled])
    train_idx, test_idx, val_idx = scaffold_split(dataset, sizes=sizes)
    return train_idx, test_idx, val_idx