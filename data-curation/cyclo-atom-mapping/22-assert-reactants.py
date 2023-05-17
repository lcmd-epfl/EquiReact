#!/usr/bin/env python3

import sys
from utils import load_graphs

tsfile = sys.argv[1]
r0file = sys.argv[2]
r1file = sys.argv[3]

print(tsfile)
load_graphs(tsfile, r0file, r1file, verbose=True)
