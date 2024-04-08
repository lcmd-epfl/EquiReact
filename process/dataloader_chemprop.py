from chemprop.data.utils import get_data_from_smiles
import pandas as pd
from process.scaffold import scaffold_split
import numpy as np
import re


def remove_atom_map_number_manual(smiles):
    smi = re.sub(':[0-9]+', '', smiles)
    return smi


def get_reactant_from_reaction_smi(reaction_smiles):
    return reaction_smiles.split('>>')[0]


def get_scaffold_splits(dataset, indices=None, sizes=(0.8, 0.1, 0.1)):
    csv_files = {'gdb': 'data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv',
                 'cyclo': 'data/cyclo/cyclo.csv',
                 'proparg': 'data/proparg/proparg.csv'}
    df = pd.read_csv(csv_files[dataset], index_col=0)
    if dataset == 'gdb':
        rsmiles = df['rsmi'].to_numpy()
    elif dataset == 'cyclo' or dataset == 'proparg':
        rsmiles = df['rxn_smiles'].apply(get_reactant_from_reaction_smi).to_numpy()

    if indices is None:
        indices = np.arange(len(rsmiles))
    rsmiles = rsmiles[indices]
    rsmiles = np.array([remove_atom_map_number_manual(smiles) for smiles in rsmiles])

    chemprop_dataset = get_data_from_smiles([[x] for x in rsmiles])
    train_idx, test_idx, val_idx = scaffold_split(chemprop_dataset, sizes=sizes, balanced=True)
    return indices[train_idx], indices[test_idx], indices[val_idx]
