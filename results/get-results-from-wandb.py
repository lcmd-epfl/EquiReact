#!/usr/bin/env python3

import warnings
from collections import defaultdict
import numpy as np
import wandb


api = wandb.Api()

for dataset in ('gdb', 'proparg'):

    runs = api.runs(f'nequireact-{dataset}')
    results = defaultdict(list)

    for run in runs:
        if run.name.startswith('cv10-LP'):
            test = run.summary['test_score']
            print(run.id, run.name, run.config['seed'], test)
            results[run.name].append(test)
    print()

    for i, ri in results.items():
        if len(ri)!=10:
            warnings.warn(f"set '{i}' contains {len(ri)} runs")
        ri = np.array(ri)
        print(i, np.mean(ri), np.std(ri))
    print()
    print()
