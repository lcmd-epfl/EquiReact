import os
import re
import numpy as np
import pandas as pd
from chemprop.data.utils import get_data_from_smiles
from process.scaffold import scaffold_split
from rdkit import Chem

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


def get_n_atoms(smiles):
    """helper function for get_size_splits:
    get number of heavy atoms from smiles string using rdkit"""
    # MolFromSmiles will by default not count Hs (implicit)
    # which is desired behaviour for following count
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('mol is none, cannot count nats')
    count = 0
    for atom in mol.GetAtoms():
        count += 1

    return count

def get_size_splits(df, dataset, indices, tr_frac, te_frac):
    """train-test split based on molecule size:
    train on smaller molecules and test on larger molecules
    (based on reactant/product depending on whether there is 1
    reactant or 1 product)

    Args:
        df: pandas df with info
        dataset: one of 'cyclo', 'gdb', 'proparg'
        indices: for subset of data (shuffling here is redundant)
        tr_frac: float
        te_frac: float

    Returns:
        tr_indices, te_indices, val_indices: tuple of list/arr of indices
        """

    df = df.reset_index(drop=True) # needed for indexing, in some dfs indices are not continous from 0
    df = df.loc[indices]

    val_frac = 1 - tr_frac - te_frac
    assert val_frac >= 0, f'val size not valid with {tr_frac=} {te_frac=}'
    if dataset == 'gdb':
        rsmiles = df['rsmi']
    elif dataset == 'cyclo':
        rsmiles = df['rxn_smiles'].apply(get_product_from_reaction_smi)
        df['rsmi'] = rsmiles
    elif dataset == 'proparg':
        rsmiles = df['rxn_smiles'].apply(get_reactant_from_reaction_smi)
        df['rsmi'] = rsmiles

    df['counts'] = rsmiles.apply(get_n_atoms)

    # sort by molecular size of rsmiles
    df = df.sort_values('counts') # increasing

    tr_size = int(len(df) * tr_frac)
    val_size = int(len(df) * val_frac)
    te_size = int(len(df) * te_frac)

    # now tr / te / val (te should be largest molecules)
    df_train_val = df[:tr_size+val_size]
    train_val_indices = df_train_val.index.to_numpy()
    df_test = df[tr_size+val_size:]
    te_indices = df_test.index.to_numpy()

    # shuffle for validation (otherwise only validate on larger mols)
    sh_indices = np.arange(len(train_val_indices))
    np.random.shuffle(sh_indices)
    train_val_indices = train_val_indices[sh_indices]
    tr_indices = train_val_indices[:tr_size]
    val_indices = train_val_indices[tr_size:]

    assert len(tr_indices) + len(val_indices) + len(te_indices) == len(df), 'data loss!'

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

    if splitter in ['scaffold', 'yasc', 'ydesc', 'msize']:
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

    elif splitter == 'msize':
        print("Splitting based on molecular size")
        tr_indices, te_indices, val_indices = get_size_splits(df, dataset, indices, tr_frac, te_frac)

    return tr_indices, te_indices, val_indices, indices
