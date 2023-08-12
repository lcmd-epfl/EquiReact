import pandas as pd

data_path = '../../data/rgd1/RGD1CHNO_smiles.csv'
x = pd.read_csv(data_path, index_col=0)
rs = x['reactant'].to_list()
ps = x['product'].to_list()

x.drop(axis=1, inplace=True, labels=set(x.columns.values)-{'DE_F'})
x['rxn_smiles'] = [r+'>>'+p for r, p in zip(rs, ps)]
x.to_csv('rgd.csv')
