#!/usr/bin/env python3

import sys
import numpy as np
from utils import load_graphs, match, get_peturbation

tsfile = sys.argv[1]
r0file = sys.argv[2]
r1file = sys.argv[3]
p0file = sys.argv[4]
p1file = sys.argv[5]

G_r0, G_r1, components = load_graphs(tsfile, r0file, r1file)
ret0, iso0 = match(G_r0, components[0])
ret1, iso1 = match(G_r1, components[1])
assert ret0
assert ret1
print(tsfile, ret0, ret1)

d0 = get_peturbation(next(iso0), G_r0)
d1 = get_peturbation(next(iso1), G_r1)
assert np.all(np.sort(np.concatenate((d0, d1)))==np.arange(G_r0.number_of_nodes()+G_r1.number_of_nodes()))

np.savetxt(p0file, d0, fmt='%d')
np.savetxt(p1file, d1, fmt='%d')
