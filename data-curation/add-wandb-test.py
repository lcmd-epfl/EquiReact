#!/usr/bin/env python3

import sys
import wandb

with open(sys.argv[1]) as f:
    strings = f.readlines()

y = [*filter(lambda x: x.startswith('wandb:  View run at'), strings)]
if len(y)!=1:
    print(y)
    raise RuntimeError
run_id = y[0].split('/')[-1].strip()

y = [*filter(lambda i: strings[i].startswith('Statistics on test'), range(len(strings)))]
if len(y)!=1:
    print(y)
    exit(0)
test_score = float(strings[y[0]+1].split()[1])

api = wandb.Api()
run = api.run(f'nequireact-gdb/{run_id}')
if not 'test_score' in run.summary and not run.state=='running':
    print(run.state, run.id, run.name, test_score)
    run.summary['test_score'] = test_score
    run.summary.update()
