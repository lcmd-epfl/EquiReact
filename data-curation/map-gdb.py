#!/usr/bin/env python3

import numpy as np
import pandas as pd
from rxnmapper import RXNMapper
from operator import itemgetter

data = pd.read_csv('../data/gdb7-22-ts/ccsdtf12_dz_cleaned_nomap.csv', index_col=0)
data.drop(axis=1, inplace=True, labels=set(data.columns.values)-{'Unnamed: 0', 'idx', 'dE0', 'rxn_smiles'})
reactions = data['rxn_smiles'].to_list()

rxn_mapper = RXNMapper()
results = map(lambda x: rxn_mapper.get_attention_guided_atom_maps([x], canonicalize_rxns=False)[0], reactions)
reactions, confidence = zip(*map(itemgetter('mapped_rxn', 'confidence'), results))

print(np.mean(list(confidence)))

data['rxn_smiles'] = list(reactions)
data.to_csv('rxnmapper.csv')
