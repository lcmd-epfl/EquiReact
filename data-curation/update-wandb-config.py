#!/usr/bin/env python3

import sys
import json
import wandb
sys.path.insert(0, '../')
import train

api = wandb.Api()
runs = api.runs('nequireact-gdb')
for run in runs:

    if not bool(run.config):
        print(run.id, run.name)

        meta = json.load(run.file("wandb-metadata.json").download(replace=True))
        argstr = meta['args']
        argstr = list(filter(lambda x: x.lower()!='true', argstr))
        for i, a in enumerate(argstr):
            if a[-5:].lower()=='=true':
                argstr[i] = a[:-5]

        args, arg_groups = train.parse_arguments(argstr)
        hyper = vars(arg_groups['hyperparameters'])

        for key, val in hyper.items():
            run.config[key] = val
        run.update()
