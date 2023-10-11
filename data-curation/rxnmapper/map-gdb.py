#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper
from operator import itemgetter

data = pd.read_csv('../data/gdb7-22-ts/ccsdtf12_dz_cleaned_nomap.csv', index_col=0)
data.drop(axis=1, inplace=True, labels=set(data.columns.values)-{'Unnamed: 0', 'idx', 'dE0', 'rxn_smiles'})
reactions = data['rxn_smiles'].to_list()

rxn_mapper = RXNMapper()
results = [rxn_mapper.get_attention_guided_atom_maps([r], canonicalize_rxns=True)[0] for r in tqdm(reactions)]
reactions, confidence = zip(*map(itemgetter('mapped_rxn', 'confidence'), results))

#np.savetxt('vcdvfvrf', np.array(list(confidence)))
#exit(0)

print(np.mean(list(confidence)))

data['rxn_smiles'] = list(reactions)
data.to_csv('rxnmapper.csv')
