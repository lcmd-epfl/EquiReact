import pandas as pd

x = pd.read_csv('rxnmapper.csv', index_col=0)
y = pd.read_csv('ccsdtf12_dz_cleaned.csv')
x['dHrxn298'] = y['dHrxn298'].to_numpy()
x.to_csv('rxnmapper2.csv')
