import numpy as np
import pandas as pd

def plot_plot(indices, pred1, pred2, outfile):
    targets = targets_all[indices]
    err1 = (pred1-targets)
    err2 = (pred2-targets)
    rmsd = np.loadtxt('rmsd.dat', usecols=2)[indices]
    labels = np.vstack((rmsd, targets)).T
    xy = np.vstack((rmsd, abs(err2)-abs(err1))).T
    np.savetxt(f'{outfile}_forplot.dat', xy)

data_dir = '../../data/cyclo'
df = pd.read_csv(f'{data_dir}/cyclo.csv')
df = df[df.bad_xtb==0].reset_index(drop=True)
targets_all = df['G_act'].values

dat1 = '../by_mol/cv10-cyclo-inv-random-noH-sub-true-ns64-nv48-d48-l2-energy-diff-node.123.dat'
dat2 = '../by_mol/cv10-cyclo-inv-random-noH-xtb-true-ns64-nv48-d48-l2-energy-diff-node.123.dat'
indices = np.loadtxt(dat1, usecols=0, dtype=int)
pred1 = np.loadtxt(dat1, usecols=2)*np.std(targets_all)+np.mean(targets_all)
pred2 = np.loadtxt(dat2, usecols=2)*np.std(targets_all)+np.mean(targets_all)
plot_plot(indices, pred1, pred2, 'cyclo_3dreact')

dat1 = '../../baseline_slatm/by_mol/slatm_10_fold_cyclo_sub_split_random.predictions.0.txt'
dat2 = '../../baseline_slatm/by_mol/slatm_10_fold_cyclo_xtb_split_random.predictions.0.txt'
indices = np.loadtxt(dat1, usecols=0, dtype=int, delimiter=',', converters=lambda x: x[1:])
pred1 = np.loadtxt(dat1, usecols=1, delimiter=',', converters=lambda x: x[:-1])
pred2 = np.loadtxt(dat2, usecols=1, delimiter=',', converters=lambda x: x[:-1])
plot_plot(indices, pred1, pred2, 'cyclo_slatm')
