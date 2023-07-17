#!/usr/bin/env python3

import pandas as pd
from rxnmapper import RXNMapper

data = pd.read_csv('../data/gdb7-22-ts/ccsdtf12_dz_cleaned_nomap.csv', index_col=0)
data.drop(axis=1, inplace=True, labels=set(data.columns.values)-{'Unnamed: 0', 'idx', 'dE0', 'rxn_smiles'})
reactions = data['rxn_smiles'].to_list()

rxn_mapper = RXNMapper()
reactions = map(lambda x: rxn_mapper.get_attention_guided_atom_maps([x], canonicalize_rxns=False), reactions)
reactions = map(lambda x: x[0]['mapped_rxn'], reactions)

data['rxn_smiles'] = list(reactions)
data.to_csv('rxnmapper.csv')
