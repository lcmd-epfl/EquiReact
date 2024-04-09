#!/usr/bin/env python3

import sys
import argparse as ap
from itertools import compress
import chemprop


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
    parser.add_argument('--scaffold',  action='store_true', help='use scaffold splits (random otherwise)')
    parser.add_argument('--withH',     action='store_true', help='use explicit H')
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":

    parser, args = argparse()
    if args.cyclo:
        data_path = '../../../data/cyclo/cyclo.csv'
        target_columns = 'G_act'
    elif args.gdb:
        data_path = '../../csv/gdb.csv'
        target_columns = 'dE0'
    elif args.proparg:
        data_path = '../../csv/proparg-fixarom.csv'
        target_columns = "Eafw"

    if args.rxnmapper:
        if args.withH:
            smiles_columns  ='rxn_smiles_rxnmapper_full'
        else:
            smiles_columns = 'rxn_smiles_rxnmapper'
    elif args.true:
        smiles_columns = 'rxn_smiles_mapped'
    elif args.none:
        smiles_columns = 'rxn_smiles'

    dataset = next(compress(('cyclo', 'gdb', 'proparg'), (args.cyclo, args.gdb, args.proparg)))
    config_path = f'../../data/hypers_{dataset}_cgr.json'

    arguments = [
        "--data_path", data_path,
        "--dataset_type",  "regression",
        "--target_columns", target_columns,
        "--smiles_columns", smiles_columns,
        "--split_sizes", "0.8", "0.1", "0.1",
        "--metric", "mae",
        "--epochs", "300",
        "--reaction",
        "--config_path", config_path,
        "--num_folds",  "10",
        "--batch_size", "50",
        "--save_dir", "./"]
    if args.scaffold:
        arguments.extend(('--split_type', 'scaffold_balanced'))
    if args.withH:
        arguments.append('--explicit_h')

    args_chemprop = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args_chemprop, train_func=chemprop.train.run_training)
    print("Mean score", mean_score, "std_score", std_score)
