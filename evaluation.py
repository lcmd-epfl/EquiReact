#!/usr/bin/env python3

import os
import sys
import re
import argparse
from types import SimpleNamespace
from datetime import datetime
from getpass import getuser
import numpy as np
import train


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  required=True,             type=str, help='path to the checkpoint log file')
parser.add_argument('--logdir',      default='logs/evaluation', type=str, help='dir for the new log file')
script_args = parser.parse_args()

run_dir = script_args.logdir
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
logname = f'{datetime.now().strftime("%y%m%d-%H%M%S.%f")}-{getuser()}'
logpath = os.path.join(run_dir, f'{logname}.log')
print(f"stdout/stderr to {logpath}")
sys.stdout = train.Logger(logpath=logpath, syspart=sys.stdout)
sys.stderr = train.Logger(logpath=logpath, syspart=sys.stderr)

with open(script_args.checkpoint, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if re.search('input args Namespace', line):
            args = eval(line.strip().replace('input args Namespace', 'SimpleNamespace'))
            break
    for line in lines:
        if re.search('and the best mae was in', line):
            best_epoch = int(line.split()[-1])-1
    for line in lines:
        if re.search('Mean MAE across splits', line):
            mae_logged = float(line.split()[-3])

print(args)
print()
args.logdir             = script_args.logdir
args.experiment_name    = None
args.wandb_name         = None
args.num_epochs         = best_epoch
args.checkpoint         = script_args.checkpoint.replace('.log', '.best_checkpoint.pt')
args.eval_on_test_split = True
if 'semiempirical' not in vars(args):
    args.semiempirical = False
print(args)
print()

maes = train.train(run_dir, logname, None, None, {}, seed=args.seed,
                   device=args.device, num_epochs=args.num_epochs, checkpoint=args.checkpoint,
                   subset=args.subset, dataset=args.dataset, process=args.process,
                   verbose=args.verbose, radius=args.radius, max_neighbors=args.max_neighbors, sum_mode=args.sum_mode,
                   n_s=args.n_s, n_v=args.n_v, n_conv_layers=args.n_conv_layers, distance_emb_dim=args.distance_emb_dim,
                   graph_mode=args.graph_mode, dropout_p=args.dropout_p, random_baseline=args.random_baseline,
                   combine_mode=args.combine_mode, atom_mapping=args.atom_mapping, CV=args.CV, attention=args.attention,
                   noH=args.noH, two_layers_atom_diff=args.two_layers_atom_diff, rxnmapper=args.rxnmapper, reverse=args.reverse,
                   xtb=args.xtb, semiempirical=args.semiempirical,
                   splitter=args.splitter,
                   split_complexes=args.split_complexes, lr=args.lr, weight_decay=args.weight_decay,
                   eval_on_test_split=args.eval_on_test_split, sweep=True)

print(f'delta MAE: {abs(mae_logged-np.mean(maes)):.2e}')
