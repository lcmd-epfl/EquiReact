#!/usr/bin/env python3

import re
from operator import itemgetter
import numpy as np
import pandas as pd
from tqdm import tqdm
import rxnmapper
from rdkit import Chem


def reset_smiles_mapping(x):
    return re.sub(':[0-9]+', '', x)


def add_h_to_smiles(reaction):
    new_smi = []
    r12, p = reaction.split('>>')
    r1, r2 = r12.split('.')
    for smi in (r1, r2, p):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        new_smi.append(Chem.MolToSmiles(mol))
    return f'{new_smi[0]}.{new_smi[1]}>>{new_smi[2]}'


def map_reactions(reactions, canonicalize_rxns=False):
    rxn_mapper = rxnmapper.RXNMapper()
    results = [rxn_mapper.get_attention_guided_atom_maps([r], canonicalize_rxns=canonicalize_rxns)[0] for r in tqdm(reactions)]
    reactions, confidence = zip(*map(itemgetter('mapped_rxn', 'confidence'), results))
    return reactions, np.array(confidence)


def map_cyclo():
    data = pd.read_csv('../../data/cyclo/cyclo.csv', index_col=0)
    reactions = data['rxn_smiles_mapped'].to_list()
    reactions = [*map(reset_smiles_mapping, reactions)]
    reactions_h = [add_h_to_smiles(r) for r in reactions]
    for output_column, rxns, canonicalize_rxns in (('rxn_smiles_rxnmapper', reactions, True),
                                                   ('rxn_smiles_rxnmapper_full', reactions_h, False)):
        mapped_rxns, confidence = map_reactions(rxns, canonicalize_rxns=canonicalize_rxns)
        print(confidence.mean())
        print()
        data[output_column] = mapped_rxns
    data.to_csv('cyclo.csv')


def map_gdb():
    data = pd.read_csv('../../data/gdb7-22-ts/ccsdtf12_dz_cleaned_nomap.csv', index_col=0)
    data.drop(axis=1, inplace=True, labels=set(data.columns.values)-{'Unnamed: 0', 'idx', 'dE0', 'rxn_smiles'})
    reactions = data['rxn_smiles'].to_list()
    output_column = 'rxn_smiles'
    for output_file, canonicalize_rxns in (('rxnmapper.csv',     False),
                                           ('rxnmapper-noH.csv', True)):
        mapped_rxns, confidence = map_reactions(reactions, canonicalize_rxns=canonicalize_rxns)
        print(confidence.mean())
        print()
        data[output_column] = mapped_rxns
        data.to_csv(output_file)


def main():
    assert rxnmapper.__version__ == '0.3.0'
    print('''warning: RXNMapper sorts the participating molecules. This behavior corrupts the input.')
        Run
          $ patch < rxnmapper.patch
        to apply the patch that discable sorting, and
          $ patch -R < rxnmapper.patch
        to revert.
    ''')
    map_cyclo()
    map_gdb()


if __name__=='__main__':
    main()
