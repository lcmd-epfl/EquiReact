import numpy as np

errors_file = '../../results/by_mol/cv10-LP-gdb-ns64-nv64-d48-layers3-vector-diff-node-truemapping.123.dat'
sign_file = 'signatures.dat'

idx = np.loadtxt(errors_file, usecols=0, dtype=int)
err = np.loadtxt(errors_file, usecols=(1,2)).T
err = err[1]-err[0]

classes = np.loadtxt(sign_file, dtype=str, delimiter=',')

class_unique, class_index, class_counts = np.unique(classes, return_inverse=True, return_counts=True)

biggest_classes = np.argsort(-class_counts)[:5]

print('class indices', biggest_classes)
print('class counts', class_counts[biggest_classes]   )

belongs_to_biggest_classes_mask = (class_index==biggest_classes[:,None]).sum(axis=0)
belongs_to_biggest_classes = np.where(belongs_to_biggest_classes_mask==0)

class_index[belongs_to_biggest_classes] = -1
classes[belongs_to_biggest_classes] = '-'

np.savetxt('class_indices.dat', class_index, fmt='%d')
