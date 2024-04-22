#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from rdkit import Chem

df = pd.read_csv('../../data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv')

reactions = df['rxn_smiles'].values

for skip_H2 in [True, False]:
    sign_file = 'signatures-skip-H2.dat' if skip_H2 else 'signatures.dat'
    signatures = []

    with open(sign_file, 'w') as f:
        for reaction in reactions:
            connectivity = []
            for smi in reaction.split('>>'):
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                Chem.SanitizeMol(mol)

                atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])
                atom_map = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])-1
                atoms_reordered = atoms[np.argsort(atom_map)]

                x = set([tuple(sorted((atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()]))) for bond in mol.GetBonds()])
                connectivity.append(x)

            signature = []
            for status, bonds in zip(('+', '-'), (connectivity[1]-connectivity[0], connectivity[0]-connectivity[1])):
                for i, j in bonds:
                    q1, q2 = atoms_reordered[i], atoms_reordered[j]
                    if skip_H2:
                        if q1=='H' or q2=='H':
                            continue
                    signature.append(status+'-'.join(sorted((q1, q2))))


            signature = tuple(sorted(signature))
            if len(signature)==0:
                signature = ('None',)
            print(*signature, file=f)
            signatures.append(signature)


    print(f'{len(set(signatures))}/{len(signatures)}', file=sys.stderr)
