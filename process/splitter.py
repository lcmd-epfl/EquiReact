import os
import re
import numpy as np
import pandas as pd
from chemprop.data.utils import get_data_from_smiles
from process.scaffold import scaffold_split


def remove_atom_map_number_manual(smiles):
    smi = re.sub(':[0-9]+', '', smiles)
    return smi


def get_reactant_from_reaction_smi(reaction_smiles):
    return reaction_smiles.split('>>')[0]


def get_product_from_reaction_smi(reaction_smiles):
    return reaction_smiles.split('>>')[1]


def get_scaffold_splits(df, dataset, indices=None, sizes=(0.8, 0.1, 0.1)):
    if dataset == 'gdb':
        rsmiles = df['rsmi'].to_numpy()
    elif dataset == 'cyclo':
        rsmiles = df['rxn_smiles'].apply(get_product_from_reaction_smi).to_numpy()
    elif dataset == 'proparg':
        rsmiles = df['rxn_smiles'].apply(get_reactant_from_reaction_smi).to_numpy()

    if indices is None:
        indices = np.arange(len(rsmiles))
    rsmiles = rsmiles[indices]
    rsmiles = np.array([remove_atom_map_number_manual(smiles) for smiles in rsmiles])

    chemprop_dataset = get_data_from_smiles([[x] for x in rsmiles])
    train_idx, test_idx, val_idx = scaffold_split(chemprop_dataset, sizes=sizes, balanced=True)
    return indices[train_idx], indices[test_idx], indices[val_idx]


def get_y_splits(df, dataset, splitter, indices, tr_size, te_size):
    y_column = {'gdb': 'dE0', 'cyclo': 'G_act', 'proparg': 'Eafw'}
    y = df[y_column[dataset]].to_numpy()
    idx4idx = np.argsort(y[indices])
    if splitter == 'ydesc':
        idx4idx = idx4idx[::-1]
    indices = indices[idx4idx]
    tr_indices, val_indices, te_indices = np.split(indices, [tr_size, tr_size+te_size])
    np.random.shuffle(tr_indices)
    np.random.shuffle(te_indices)
    np.random.shuffle(val_indices)
    return tr_indices, te_indices, val_indices


def split_dataset(nreactions, splitter, tr_frac, dataset, subset=None):
    # 1) seed `np.random` and `random` before calling this fn
    # 2) use the output indices with np.arrays, lists, df.iloc[]
    indices = np.arange(nreactions)
    len_before = len(indices)
    np.random.shuffle(indices)
    len_after = len(indices)
    assert len_before == len_after, "lost data in shuffle"
    if subset:
        indices = indices[:subset]
        assert len(indices) == subset, "lost data in subset"

    te_frac = (1. - tr_frac) / 2
    tr_size = round(tr_frac * len(indices))
    te_size = round(te_frac * len(indices))
    va_size = len(indices) - tr_size - te_size

    if splitter in ['scaffold', 'yasc', 'ydesc']:
        csv_files = {'gdb': 'data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv',
                     'cyclo': 'data/cyclo/cyclo.csv',
                     'proparg': 'data/proparg/proparg.csv'}
        dirname = os.path.abspath(f'{os.path.dirname(__file__)}/../')
        df = pd.read_csv(f'{dirname}/{csv_files[dataset]}', index_col=0)

    if splitter == 'random':
        print("Using random splits")
        tr_indices, te_indices, val_indices = np.split(indices, [tr_size, tr_size+te_size])

    elif splitter in ['yasc', 'ydesc']: # splits based on the target value
        print(f"Using target-based splits ({'ascending' if splitter=='yasc' else 'descending'} order)")
        tr_indices, te_indices, val_indices = get_y_splits(df, dataset, splitter, indices, tr_size, te_size)

    elif splitter == 'scaffold':
        print("Using scaffold splits")
        tr_indices, te_indices, val_indices = get_scaffold_splits(df, dataset=dataset,
                                                                  indices=indices,
                                                                  sizes=(tr_frac, 1-(tr_frac+te_frac), te_frac))
    return tr_indices, te_indices, val_indices, indices
