import numpy as np

#errors_file = '../by_mol/cv10-gdb-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat'
errors_file = '../by_mol/cv10-gdb-random-withH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat'
sign_file = 'signatures.dat'

idx = np.loadtxt(errors_file, usecols=0, dtype=int)
err = np.loadtxt(errors_file, usecols=(1,2)).T
err = err[1]-err[0]

classes = np.loadtxt(sign_file, dtype=str, delimiter=',')

class_unique, class_index, class_counts = np.unique(classes, return_inverse=True, return_counts=True)

biggest_classes = np.argsort(-class_counts)[:5]

belongs_to_biggest_classes_mask = (class_index==biggest_classes[:,None]).sum(axis=0)
belongs_to_biggest_classes = np.where(belongs_to_biggest_classes_mask==0)

classes[belongs_to_biggest_classes] = '-'
class_test = classes[idx]

idx_test = np.where(class_test!='-')
class_test = class_test[idx_test]
err = err[idx_test]

class_unique_test, idx_new = np.unique(class_test, return_inverse=True)

for i in range(5):
    print('#', class_unique_test[i])
    for j in err[idx_new==i]:
        print(j)
    print()
    print()
