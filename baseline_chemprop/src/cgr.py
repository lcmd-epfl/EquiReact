#!/usr/bin/env python3

import os
import sys
import argparse as ap
from itertools import compress
import random
import numpy as np
import pandas as pd
import chemprop

chempropdir = os.path.abspath(f'{os.path.dirname(__file__)}/../')
rootdir = os.path.abspath(f'{chempropdir}/../')
sys.path.insert(0, rootdir)
from process.splitter import split_dataset


def argparse():
    parser = ap.ArgumentParser()
    g1 = parser.add_mutually_exclusive_group(required=True)
    g1.add_argument('--true',          action='store_true', help='use true atom mapping')
    g1.add_argument('--rxnmapper',     action='store_true', help='use atom mapping from rxnmapper')
    g1.add_argument('--none',          action='store_true', help='use non-mapped smiles')
    g2 = parser.add_mutually_exclusive_group(required=True)
    g2.add_argument('-c', '--cyclo',   action='store_true', help='use curated Cyclo-23-TS dataset')
    g2.add_argument('-p', '--proparg', action='store_true', help='use Proparg-21-TS dataset with fragment-based SMILES')
    g2.add_argument('-g', '--gdb',     action='store_true', help='use curated GDB7-22-TS dataset')
    g3 = parser.add_mutually_exclusive_group(required=False)
    g3.add_argument('--scaffold',      action='store_true', help='use scaffold splits (random otherwise)')
    g3.add_argument('--yasc',          action='store_true', help='use yasc splits (random otherwise)')
    g3.add_argument('--ydesc',         action='store_true', help='use ydesc splits (random otherwise)')
    g3.add_argument('--sizeasc',       action='store_true', help='use sizeasc splits (random otherwise)')
    g3.add_argument('--sizedesc',      action='store_true', help='use sizedesc splits (random otherwise)')
    parser.add_argument('--withH',     action='store_true', help='use explicit H')
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":
    parser, args = argparse()

    if args.proparg:
        assert chemprop.__version__ == '1.5.0'
        print(f'''warning: rdkit does not like hypervalent Si. The script may fail for the proparg dataset.')
            Run
              $ patch < {chempropdir}/src/chemprop.patch
            to apply the patch that disables the valence check, and
              $ patch -R < {chempropdir}/src/chemprop.patch
            to revert.
        ''')

    if args.cyclo:
        data_path = 'data/cyclo/cyclo.csv'
        target_columns = 'G_act'
    elif args.gdb:
        data_path = 'data/gdb7-22-ts/gdb.csv'
        target_columns = 'dE0'
    elif args.proparg:
        data_path = 'data/proparg/proparg.csv'
        target_columns = "Eafw"

    if args.rxnmapper:
        if args.proparg:
            raise RuntimeError('no rxnmapper for proparg')
        if args.withH:
            smiles_columns  ='rxn_smiles_rxnmapper_full'
        else:
            smiles_columns = 'rxn_smiles_rxnmapper'
    elif args.true:
        smiles_columns = 'rxn_smiles_mapped'
    elif args.none:
        smiles_columns = 'rxn_smiles'

    dataset = next(compress(('cyclo', 'gdb', 'proparg'), (args.cyclo, args.gdb, args.proparg)))
    config_path = f'{chempropdir}/config/hypers_{dataset}_cgr.json'


    CV = 10
    tr_frac = 0.8

    if args.scaffold:
        splitter = 'scaffold'
    elif args.yasc:
        splitter = 'yasc'
    elif args.ydesc:
        splitter = 'ydesc'
    elif args.sizeasc:
        splitter = 'sizeasc'
    elif args.sizedesc:
        splitter = 'sizedesc'
    else:
        splitter = 'random'

    seed = 123
    scores = np.zeros(CV)
    for i in range(CV):
        print(f"CV iter {i+1}/{CV}")

        np.random.seed(seed)
        random.seed(seed)

        data = pd.read_csv(f'{rootdir}/{data_path}')

        tr_indices, te_indices, val_indices, _ = split_dataset(nreactions=len(data), splitter=splitter,
                                                               tr_frac=tr_frac, dataset=dataset, subset=None)
        data.iloc[tr_indices].to_csv(f'data_{seed}_train.csv')
        data.iloc[te_indices].to_csv(f'data_{seed}_test.csv')
        data.iloc[val_indices].to_csv(f'data_{seed}_val.csv')

        arguments = [
            "--data_path",           f'data_{seed}_train.csv',
            "--separate_val_path",   f'data_{seed}_val.csv',
            "--separate_test_path",  f'data_{seed}_test.csv',
            "--dataset_type",        "regression",
            "--target_columns",      target_columns,
            "--smiles_columns",      smiles_columns,
            "--metric",              "mae",
            "--extra_metrics",       "rmse",
            "--epochs",              "300",
            "--reaction",
            "--config_path",         config_path,
            "--batch_size",          "50",
            "--save_dir",            f"fold_{i}"]
        if args.withH:
            arguments.append('--explicit_h')

        args_chemprop = chemprop.args.TrainArgs().parse_args(arguments)
        score, _ = chemprop.train.cross_validate(args=args_chemprop, train_func=chemprop.train.run_training)
        scores[i] = score

        seed += 1

    print("Mean score", scores.mean(), "std_score", scores.std())
