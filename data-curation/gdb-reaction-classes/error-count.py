import numpy as np

errors_file = '../../results/by_mol/cv10-LP-gdb-ns64-nv64-d48-layers3-vector-diff-node-truemapping.123.dat'
sign_file = 'signatures.dat'

idx = np.loadtxt(errors_file, usecols=0, dtype=int)
err = np.loadtxt(errors_file, usecols=(1,2)).T
err = err[1]-err[0]

classes = np.loadtxt(sign_file, dtype=str, delimiter=',')

class_unique, class_index, class_counts = np.unique(classes, return_inverse=True, return_counts=True)

class_counts_full = class_counts[class_index]

class_test = classes[idx]
class_counts_test = class_counts_full[idx]

for x,y in zip(err, class_counts_test):
    print(abs(x),y)
