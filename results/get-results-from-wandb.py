#!/usr/bin/env python3

import warnings
from collections import defaultdict
import numpy as np
import wandb


api = wandb.Api()

for dataset in ('proparg', 'cyclo', 'gdb'):

    runs = api.runs(f'nequireact-{dataset}-80')
    results = defaultdict(list)

    for run in runs:
        if run.state=='running':
            continue
        if run.name.startswith('lc-cv10-') or run.name.startswith('cv10-'):
            test_score = run.summary['test_score']
            test_rmse  = run.summary['test_rmse']
            print(run.id, run.name, run.config['seed'], test_score, test_rmse)
            results[run.name].append((test_score, test_rmse))
    print()

    for i, ri in results.items():
        if len(ri)!=10:
            warnings.warn(f"set '{i}' contains {len(ri)} runs")
        ri = np.array(ri)
        test_mean = np.mean(ri, axis=0)
        test_std  = np.std(ri, axis=0)
        print(i, test_mean[0], test_std[0], test_mean[1], test_std[1])
    print()
    print()
