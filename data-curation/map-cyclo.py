#!/usr/bin/env python3

import re
from operator import itemgetter
import numpy as np
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper

def reset_smiles_mapping(x):
    return re.sub(':[0-9]+', '', x)

data = pd.read_csv('../data/cyclo/full_dataset_2.csv', index_col=0)
#data.drop(axis=1, inplace=True, labels=set(data.columns.values)-{'Unnamed: 0', 'idx', 'dE0', 'rxn_smiles'})

reactions = data['rxn_smiles_mapped'].to_list()
reactions = [*map(reset_smiles_mapping, reactions)]

rxn_mapper = RXNMapper()
results = [rxn_mapper.get_attention_guided_atom_maps([r], canonicalize_rxns=False)[0] for r in tqdm (reactions)]
reactions, confidence = zip(*map(itemgetter('mapped_rxn', 'confidence'), results))

print(np.mean(list(confidence)))

data['rxn_smiles_rxnmapper'] = list(reactions)
data.to_csv('rxnmapper.csv')
