#!/usr/bin/env python3

import wandb

api = wandb.Api()
runs = api.runs('nequireact-gdb')
for run in runs:
    if not 'val_score' in run.summary and not run.state=='running':
        print(run.state, run.id, run.name)
        last_val = [_ for _ in run.scan_history(keys=['val_loss', 'val_score'])][-1]
        run.summary['val_loss'] = last_val['val_loss']
        run.summary['val_score'] = last_val['val_score']
        run.summary.update()
    if not 'val_score_best' in run.summary and not run.state=='running':
        print(run.state, run.id, run.name)
        val_score_best = min(x['val_score'] for x in run.scan_history(keys=['val_score']))
        run.summary['val_score_best'] = val_score_best
        run.summary.update()
