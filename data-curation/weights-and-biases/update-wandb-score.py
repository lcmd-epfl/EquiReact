#!/usr/bin/env python3

import sys
import json
import wandb
sys.path.insert(0, '../')
import train

api = wandb.Api()
runs = api.runs('nequireact-gdb')
for run in runs:
    if not 'val_score' in run.summary:
        print(run.id, run.name)
        last_val = [_ for _ in run.scan_history(keys=['val_loss', 'val_score'])][-1]
        run.summary['val_loss'] = last_val['val_loss']
        run.summary['val_score'] = last_val['val_score']
        run.summary.update()
