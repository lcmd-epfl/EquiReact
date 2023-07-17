import pandas as pd

x = pd.read_csv('rxnmapper_rxns.csv', index_col=0)

y = pd.read_csv('ccsdtf12_dz_cleaned.csv')


idx = y['Unnamed: 0'].to_numpy()

x = x.iloc[idx]

x['idx'] = y['idx'].to_numpy()

x.to_csv('rxnmapper_rxns_cleaned.csv')

