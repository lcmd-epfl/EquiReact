#!/usr/bin/env python3

import os
import argparse as ap
import numpy as np
from reaction_reps import QML
from learning import predict_CV

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--database', default='gdb')
    parser.add_argument('--train_size', default=0.8)
    parser.add_argument('-xtb', '--xtb', action='store_true', default=False)
    parser.add_argument('--xtb_subset', action='store_true', default=False, help='Run on the xtb data subset (not necessarily at xtb level')
    parser.add_argument('--splitter', default='random', help='splitter random or scaffold')
    g1 = parser.add_mutually_exclusive_group(required=False)
    g1.add_argument('--no_h_total',     action='store_true', help='representation without H whatsoever')
    g1.add_argument('--no_h_atoms',     action='store_true', help='representation without H atoms but with X-H and X-X-H features')
    args = parser.parse_args()
    print("Running for database", args.database)
    if args.xtb:
        print("Using xtb geoms")
        xtb_text = '_xtb'
    else:
        xtb_text = ''

    if args.xtb_subset:
        print("Using xtb subset")
        s_text = '_sub'
    else:
        s_text = ''
    if args.no_h_total:
        h_text = '_no_h_total'
    elif args.no_h_atoms:
        h_text = '_no_h_atoms'
    else:
        h_text = ''

    database_label = args.database + xtb_text + h_text

    print(f"Using {args.splitter} splits")

    qml = QML()

    if args.database == 'gdb':
        qml.get_GDB7_ccsd_data(xtb=args.xtb, xtb_subset=args.xtb_subset)
    elif args.database == 'cyclo':
        qml.get_cyclo_data(xtb=args.xtb, xtb_subset=args.xtb_subset)
    elif args.database == 'proparg':
        qml.get_proparg_data(xtb=args.xtb)

    slatm_save = f'repr/slatm_{args.database}{xtb_text}{s_text}{h_text}.npy'
    barriers = qml.barriers

    CV = 10
    if not os.path.exists(slatm_save):
        slatm = qml.get_SLATM(no_h_atoms=args.no_h_atoms, no_h_total=args.no_h_total)
        np.save(slatm_save, slatm)
        print("SLATM saved to", slatm_save)
    else:
        slatm = np.load(slatm_save)

    slatm_save = f'results/slatm_{CV}_fold_{args.database}{xtb_text}{s_text}{h_text}_split_{args.splitter}.npy'
    slatm_pred = f'by_mol/slatm_{CV}_fold_{args.database}{xtb_text}{s_text}{h_text}_split_{args.splitter}.predictions.'+'{i}'+'.txt'

    print(f'{database_label=}')
    if not os.path.exists(slatm_save):
        maes_slatm, rmses_slatm, hyperparameters = predict_CV(slatm, barriers, CV=CV, train_size=args.train_size,
                                                   save_predictions=slatm_pred,
                                                   splitter=args.splitter,
                                                   dataset=database_label, seed=123)
        np.save(slatm_save, np.array((maes_slatm, rmses_slatm, hyperparameters), dtype=object))
    else:
        print(f'reading from {slatm_save}')
        maes_slatm, rmses_slatm, hyperparameters = np.load(slatm_save, allow_pickle=True)
    print()
    print(f'"{database_label}" : {hyperparameters}')
    print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')
    print(f'slatm rmse {np.mean(rmses_slatm)} +- {np.std(rmses_slatm)}')
