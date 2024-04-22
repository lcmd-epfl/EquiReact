#!/usr/bin/env python3

import os
import numpy as np

sign_file = 'signatures.dat'
classes = np.loadtxt(sign_file, dtype=str, delimiter=',')
class_unique, class_index, class_counts = np.unique(classes, return_inverse=True, return_counts=True)
n_classes = 5
biggest_classes = np.argsort(-class_counts)[:n_classes]
belongs_to_biggest_classes_mask = (class_index==biggest_classes[:,None]).sum(axis=0)
belongs_to_biggest_classes = np.where(belongs_to_biggest_classes_mask==0)
classes[belongs_to_biggest_classes] = '-'

for hlabel in ['noH', 'withH']:
    errors_file = f'../by_mol/cv10-gdb-inv-random-{hlabel}-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat'

    idx = np.loadtxt(errors_file, usecols=0, dtype=int)
    err = np.loadtxt(errors_file, usecols=(1,2)).T
    err = err[1]-err[0]

    class_test = classes[idx]

    idx_test = np.where(class_test!='-')
    class_test = class_test[idx_test]
    err = err[idx_test]

    class_unique_test, idx_new = np.unique(class_test, return_inverse=True)

    with open(f'class-errors-{os.path.basename(errors_file)}', 'w') as f:
        for i, class_unique_test_i in enumerate(class_unique_test):
            print('#', class_unique_test_i, file=f)
            for j in err[idx_new==i]:
                print(j, file=f)
            print('\n', file=f)
