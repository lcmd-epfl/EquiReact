from glob import glob
import subprocess
import numpy as np
import pandas as pd
import ase.io
import rmsd   # https://github.com/charnley/rmsd

def call_rmsd(args):
    filename = f'{rmsd.__path__[0]}/calculate_rmsd.py'
    cmd = ["python", f"{filename}", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = proc.communicate()
    if stderr is not None:
        print(stderr.decode())
    return float(stdout.decode().strip().split("\n")[0])

data_dir = '../data/cyclo'

df = pd.read_csv(f'{data_dir}/cyclo.csv', index_col=0)
bad_idx = np.loadtxt(f'{data_dir}/bad-xtb.dat', dtype=int)
for idx in bad_idx:
    df.drop(df[df['rxn_id']==idx].index, axis=0, inplace=True)


print('# n_atoms n_atoms_heavy rmsd rmsd/2/n_atoms')
for idx, switch in zip(df['rxn_id'], df['switch_reactants']):
    pfile_dft  = glob(f'{data_dir}/xyz/{idx}/p*.xyz')[0]
    r0file_dft = sorted(glob(f'{data_dir}/xyz/{idx}/r1*.xyz'))[-1]
    r1file_dft = sorted(glob(f'{data_dir}/xyz/{idx}/r0*.xyz'))[-1]
    if switch:
        r0file_dft, r1file_dft = r1file_dft, r0file_dft
    pfile_xtb  = f'{data_dir}/xyz-xtb/Product_{idx}.xyz'
    r0file_xtb = f'{data_dir}/xyz-xtb/Reactant_{idx}_0.xyz'
    r1file_xtb = f'{data_dir}/xyz-xtb/Reactant_{idx}_1.xyz'

    dft_files = [pfile_dft, r0file_dft, r1file_dft]
    xtb_files = [pfile_xtb, r0file_xtb, r1file_xtb]

    diff = 0.0
    for dft, xtb in zip(dft_files, xtb_files):
        RMSD = call_rmsd(['--reorder', dft, xtb])
        diff += RMSD**2
    m = ase.io.read(pfile_dft)
    N = m.get_global_number_of_atoms()
    n = np.count_nonzero(m.get_atomic_numbers()>1)
    print(N, n, np.sqrt(diff), np.sqrt(diff)/(2*N))
