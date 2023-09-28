#!/usr/bin/env python3

import sys
import ase.io

mol = ase.io.read(sys.argv[1])
q = mol.get_atomic_numbers()
r = mol.get_positions()

print('''$system mem=64 disk=-64 $end
$control
 task=hessian
 task=optimize
 theory=qm_n3
 basis=qm.in
$end

$optimize tol=1e-4 $end

$molecule
charge=1
cart''')
for qi, ri in zip(q, r):
    print(qi, *ri)
print('$end')
