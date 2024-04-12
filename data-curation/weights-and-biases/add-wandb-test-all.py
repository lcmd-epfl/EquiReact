#!/usr/bin/env python3

import sys
import math
import wandb

with open(sys.argv[1]) as f:
    strings = f.readlines()

y = [*filter(lambda x: x.startswith('wandb:  View run at'), strings)]
if len(y)!=1:
    print(y)
    raise RuntimeError
run_id = y[0].split('/')[-1].strip()

y = [*filter(lambda x: x.startswith("In trainer, metrics is {'mae': MAE()} and std is"), strings)]
if len(y)!=1:
    print(y)
    raise RuntimeError
data_std = float(y[0].split()[-1].strip())

y = [*filter(lambda i: strings[i].startswith('Statistics on test'), range(len(strings)))]
if len(y)!=1:
    print(y)
    raise RuntimeError
test_score = float(strings[y[0]+1].split()[1])
test_mse = float(strings[y[0]+2].split()[1])

test_rmse = math.sqrt(test_mse)*data_std

api = wandb.Api()
run = api.run(f'nequireact-cyclo-80/{run_id}')
if (not 'test_rmse' in run.summary or not 'test_score' in run.summary) and not run.state=='running':
    print(run.state, run.id, run.name, test_score, test_rmse)
    if not 'test_score' in run.summary:
        run.summary['test_score'] = test_score
    if not 'test_rmse' in run.summary:
        run.summary['test_rmse'] = test_rmse
    run.summary.update()
