#!/usr/bin/env python3

import re
from operator import itemgetter
import numpy as np
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper
from rdkit import Chem

def reset_smiles_mapping(x):
    return re.sub(':[0-9]+', '', x)

data = pd.read_csv('../data/cyclo/cyclo.csv', index_col=0)

reactions = data['rxn_smiles_mapped'].to_list()
reactions = [*map(reset_smiles_mapping, reactions)]

if 0:
    # without H
    rxn_mapper = RXNMapper()
    results = [rxn_mapper.get_attention_guided_atom_maps([r], canonicalize_rxns=True)[0] for r in tqdm (reactions)]
    reactions, confidence = zip(*map(itemgetter('mapped_rxn', 'confidence'), results))
    print(np.mean(list(confidence)))
    data['rxn_smiles_rxnmapper'] = list(reactions)
else:
    # with H
    new_reactions = []
    for reaction in reactions:
        new_smi = []
        r12, p = reaction.split('>>')
        r1, r2 = r12.split('.')
        for smi in (r1, r2, p):
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)
            new_smi.append(Chem.MolToSmiles(mol))
        new_smi = new_smi[0]+'.'+new_smi[1]+'>>'+new_smi[2]
        new_reactions.append(new_smi)


    rxn_mapper = RXNMapper()
    results = [rxn_mapper.get_attention_guided_atom_maps([r], canonicalize_rxns=False)[0] for r in tqdm (new_reactions)]
    reactions, confidence = zip(*map(itemgetter('mapped_rxn', 'confidence'), results))
    print(np.mean(list(confidence)))
    data['rxn_smiles_rxnmapper_full'] = list(reactions)


data.to_csv('rxnmapper.csv')
