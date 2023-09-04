#!/usr/bin/env python3

import sys
import argparse as ap
import chemprop

# to make chemprop work with --proparg_combinat/--proparg_stereo,
# patch ${CONDA_PREFIX}/lib/python3.8/site-packages/chemprop/rdkit.py
# with cgr_proparg_patch.txt


def argparse():
    parser = ap.ArgumentParser()
    g1 = parser.add_mutually_exclusive_group(required=True)
    g1.add_argument('--true', action='store_true', help='use true atom mapping')
    g1.add_argument('--random', action='store_true', help='use random atom mapping')
    g1.add_argument('--rxnmapper', action='store_true', help='use atom mapping from rxnmapper')
    g1.add_argument('--none', action='store_true', help='use non-mapped smiles')
    g2 = parser.add_mutually_exclusive_group(required=True)
    g2.add_argument('-c', '--cyclo', action='store_true', help='use curated Cyclo-23-TS dataset')
    g2.add_argument('-p', '--proparg', action='store_true', help='use Proparg-21-TS dataset with SMILES from xyz')
    g2.add_argument('-g', '--gdb', action='store_true', help='use curated GDB7-22-TS dataset')
    g2.add_argument('-r', '--rgd', action='store_true', help='use RGD1 dataset')
    g2.add_argument('--gdb_full', action='store_true', help='use GDB7-22-TS dataset')
    g2.add_argument('--proparg_combinat', action='store_true', help='use Proparg-21-TS dataset with fragment-based SMILES')
    g2.add_argument('--proparg_stereo', action='store_true', help='use Proparg-21-TS dataset with stereochemistry-enriched fragment-based SMILES')
    g2.add_argument('--cyclo_full', action='store_true', help='use full Cyclo-23-TS dataset')
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":

    parser, args = argparse()
    if args.cyclo:
        data_path = '../../../data/cyclo/mod_dataset.csv'
        target_columns = 'G_act'
    elif args.rgd:
        data_path = '../../csv/rgd.csv'
        target_columns = 'DE_F'
    elif args.gdb:
        data_path = '../../csv/gdb.csv'
        target_columns = 'dE0'
    elif args.gdb_full:
        data_path = '../../csv/gdb-full.csv'
        target_columns = 'dE0'
    elif args.proparg:
        data_path = '../../csv/proparg.csv'
        target_columns = "Eafw"
    elif args.proparg_combinat:
        data_path = '../../csv/proparg-fixarom.csv'
        target_columns = "Eafw"
    elif args.proparg_stereo:
        data_path = '../../csv/proparg-stereo.csv'
        target_columns = "Eafw"
    if args.cyclo_full:
        data_path = '../../csv/cyclo.csv'
        target_columns = 'G_act'

    if args.random:
        smiles_columns = 'rxn_smiles_random'
    elif args.rxnmapper:
        smiles_columns = 'rxn_smiles_rxnmapper'
    elif args.true:
        smiles_columns = 'rxn_smiles_mapped'
    elif args.none:
        smiles_columns = 'rxn_smiles'

    arguments = [
        "--data_path", data_path,
        "--dataset_type",  "regression",
        "--target_columns", target_columns,
        "--smiles_columns", smiles_columns,
        "--metric", "mae",
        "--dropout", "0.05",
        "--epochs", "300",
        "--reaction",
        "--num_folds",  "10",
        "--batch_size", "50",
        "--save_dir", "./"]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    print("Mean score", mean_score, "std_score", std_score)
