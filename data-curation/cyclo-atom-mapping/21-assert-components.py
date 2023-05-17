#!/usr/bin/env python3

import sys
import networkx.algorithms.components.connected as con
from utils import *

tsfile = sys.argv[1]
G = make_graph(tsfile)
nc = con.number_connected_components(G)
if nc!=2:
    print(tsfile, nc)
