import numpy as np

sign_file = 'signatures.dat'

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
