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
    args = parser.parse_args()
    database = args.database
    xtb = args.xtb
    xtb_subset = args.xtb_subset
    splitter = args.splitter
    print("Running for database", database)
    if xtb:
        print("Using xtb geoms")
        xtb_text = '_xtb'
        database_label = database + xtb_text
    else:
        xtb_text = ''
        database_label = database

    if xtb_subset:
        print("Using xtb subset")
        s_text = '_sub'
    else:
        s_text = ''

    print(f"Using {splitter} splits")

    qml = QML()

    if database == 'gdb':
        qml.get_GDB7_ccsd_data(xtb=xtb, xtb_subset=xtb_subset)
        kernel = 'laplacian'

    elif database == 'cyclo':
        qml.get_cyclo_data(xtb=xtb, xtb_subset=xtb_subset)
        kernel = 'laplacian'

    elif database == 'proparg':
        qml.get_proparg_data(xtb=xtb)
        kernel = 'gaussian'

    slatm_save = f'repr/slatm_{database}{xtb_text}{s_text}.npy'
    barriers = qml.barriers

    CV = 10
    if not os.path.exists(slatm_save):
        slatm = qml.get_SLATM()
        np.save(slatm_save, slatm)
        print("SLATM saved to", slatm_save)
    else:
        slatm = np.load(slatm_save)

    slatm_save = f'results/slatm_{CV}_fold_{database}{xtb_text}{s_text}_split_{splitter}.npy'
    slatm_pred = f'by_mol/slatm_{CV}_fold_{database}{xtb_text}{s_text}_split_{splitter}.predictions.'+'{i}'+'.txt'

    if not os.path.exists(slatm_save):
        maes_slatm, rmses_slatm = predict_CV(slatm, barriers, CV=CV, train_size=args.train_size,
                                             save_predictions = slatm_pred,
                                             splitter=splitter, kernel=kernel,
                                             dataset=database_label, seed=123)
        np.save(slatm_save, (maes_slatm, rmses_slatm))
    else:
        maes_slatm, rmses_slatm = np.load(slatm_save)
    print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')
    print(f'slatm rmse {np.mean(rmses_slatm)} +- {np.std(rmses_slatm)}')
