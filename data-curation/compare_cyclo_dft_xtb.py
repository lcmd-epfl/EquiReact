import numpy as np
import pandas as pd
import ase
import ase.io

from glob import glob

df = pd.read_csv('../data/cyclo/cyclo.csv', index_col=0)
bad_idx = np.loadtxt('../data/cyclo/bad-xtb.dat', dtype=int)
for idx in bad_idx:
    df.drop(df[df['rxn_id']==idx].index, axis=0, inplace=True)
x = df['rxn_id'].values

for idx in x:

    pfile  = glob(f'../data/cyclo/xyz/{idx}/p*.xyz')[0]
    r0file = glob(f'../data/cyclo/xyz/{idx}/r1*.xyz')[-1]
    r1file = glob(f'../data/cyclo/xyz/{idx}/r0*.xyz')[-1]

    Pfile  = f'../data/cyclo/xyz-xtb/Product_{idx}.xyz'
    R0file = f'../data/cyclo/xyz-xtb/Reactant_{idx}_0.xyz'
    R1file = f'../data/cyclo/xyz-xtb/Reactant_{idx}_1.xyz'

    diff = 0
    for x,y in zip((pfile, r0file, r1file), (Pfile, R0file, R1file)):
        m1 = ase.io.read(pfile)
        m2 = ase.io.read(Pfile)
        diff += np.linalg.norm(m1.positions-m1.get_center_of_mass()-m2.positions+m2.get_center_of_mass())**2
    m = ase.io.read(pfile)
    N = m.get_global_number_of_atoms()
    n = np.count_nonzero(m.get_atomic_numbers()>1)
    print(N, n, np.sqrt(diff)/(2*N))
