from chemprop.data.utils import get_data_from_smiles
import pandas as pd
from process.scaffold import scaffold_split
import numpy as np
from rdkit import Chem
import re

def remove_atom_map_number(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol.AddHs()
    smi = Chem.MolToSmiles(mol)

    assert smi != '', 'empty smiles'

    mol2 = Chem.MolFromSmiles(smi)
    assert mol2 is not None, 'mol is none'
    assert mol2.GetNumHeavyAtoms() > 0, 'no atoms in mol'
    return smi

def remove_atom_map_number_manual(smiles):
    smi = re.sub(':[0-9]+', '', smiles)
    return smi

def get_scaffold_splits_gdb(data='data/gdb7-22-ts/ccsdtf12_dz.csv',
                            shuffle_indices=None,
                            reactant_col='rsmi'):

    df = pd.read_csv(data, index_col=0)
    if shuffle_indices is None:
        shuffle_indices = np.arange(len(df))
    rsmiles = df[reactant_col].to_numpy()
    rsmiles = np.array([remove_atom_map_number_manual(smiles) for smiles in rsmiles])
    # shuffle indices
    init_n = len(rsmiles)
    rsmiles = rsmiles[shuffle_indices]
    final_n = len(rsmiles)
    assert init_n == final_n, 'lost some atoms using shuffle atoms'

    # this gives a lot of warnings and smiles parse errors if we dont turn flag off
   # for smi in rsmiles:
    #    print(smi)
   #     get_data_from_smiles([smi])
    #exit()
    dataset = get_data_from_smiles(rsmiles)
    train_idx, test_idx, val_idx = scaffold_split(dataset)
    return train_idx, test_idx, val_idx

get_scaffold_splits_gdb()