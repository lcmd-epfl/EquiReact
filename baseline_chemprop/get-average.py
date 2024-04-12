#!/usr/bin/env python3

# grep 0.0 results/*/fold_?/test_scores.csv > res.txt

import numpy as np

keys = np.loadtxt('res.txt', usecols=0, dtype=str, delimiter=',')
vals = np.loadtxt('res.txt', usecols=[1,4], delimiter=',')
keys = [i.split('/')[1] for i in keys]

data = {}
for key, val in zip(keys, vals):
    if key not in data:
        data[key] = []
    data[key].append(val)

print('set mae_mean mae_std rmse_mean rmse_std')
for key, val in data.items():
    if len(val)!=10:
        print(f'!!! warning: {key} has only {len(val)} entries')
    val = np.array(val)
    val_mean = val.mean(axis=0)
    val_std  = val.std(axis=0)
    print(key, val_mean[0], val_std[0], val_mean[1], val_std[1])

