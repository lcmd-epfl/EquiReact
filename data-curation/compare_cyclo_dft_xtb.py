from glob import glob
import numpy as np
import pandas as pd
import ase
import ase.io

data_dir = '../data/cyclo'

df = pd.read_csv(f'{data_dir}/cyclo.csv', index_col=0)
bad_idx = np.loadtxt(f'{data_dir}/bad-xtb.dat', dtype=int)
for idx in bad_idx:
    df.drop(df[df['rxn_id']==idx].index, axis=0, inplace=True)
indices = df['rxn_id'].values

for idx in indices:

    pfile_dft  = glob(f'{data_dir}/xyz/{idx}/p*.xyz')[0]
    r0file_dft = sorted(glob(f'{data_dir}/xyz/{idx}/r1*.xyz'))[-1]
    r1file_dft = sorted(glob(f'{data_dir}/xyz/{idx}/r0*.xyz'))[-1]
    pfile_xtb  = f'{data_dir}/xyz-xtb/Product_{idx}.xyz'
    r0file_xtb = f'{data_dir}/xyz-xtb/Reactant_{idx}_0.xyz'
    r1file_xtb = f'{data_dir}/xyz-xtb/Reactant_{idx}_1.xyz'

    dft_files = [pfile_dft, r0file_dft, r1file_dft]
    xtb_files = [pfile_xtb, r0file_xtb, r1file_xtb]

    diff = 0.0
    for x, y in zip(dft_files, xtb_files):
        m1 = ase.io.read(pfile_dft)
        m2 = ase.io.read(pfile_xtb)
        diff += np.linalg.norm(m1.positions-m1.get_center_of_mass()-m2.positions+m2.get_center_of_mass())**2
    m = ase.io.read(pfile_dft)
    N = m.get_global_number_of_atoms()
    n = np.count_nonzero(m.get_atomic_numbers()>1)
    print(N, n, np.sqrt(diff)/(2*N))
