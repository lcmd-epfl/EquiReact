import argparse as ap
from slatm_baseline.reaction_reps import QML
from slatm_baseline.learning import predict_CV
import numpy as np
import os

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--database', default='gdb')
    parser.add_argument('-xtb', '--xtb', action='store_true', default=False)
    parser.add_argument('-s', '--xtb_subset', action='store_true', default=False, help='Run on the xtb data subset (not necessarily at xtb level')
    args = parser.parse_args()
    database = args.database
    xtb = args.xtb
    xtb_subset = args.xtb_subset
    print("Running for database", database)
    if xtb:
        print("Using xtb geoms")
        xtb_text = '_xtb'
    else:
        xtb_text = ''

    if xtb_subset:
        print("Using xtb subset")
        s_text = '_sub'
    else:
        s_text = ''

    qml = QML()

    if database == 'gdb':
        qml.get_GDB7_ccsd_data(xtb=xtb, xtb_subset=xtb_subset)

    elif database == 'cyclo':
        qml.get_cyclo_data(xtb=xtb)

    elif database == 'proparg':
        qml.get_proparg_data(xtb=xtb)

    slatm_save = f'slatm_baseline/slatm_{database}{xtb_text}{s_text}.npy'
    barriers = qml.barriers

    CV = 10
    if not os.path.exists(slatm_save):
        slatm = qml.get_SLATM()
        np.save(slatm_save, slatm)
    else:
        slatm = np.load(slatm_save)

    slatm_save = f'slatm_baseline/slatm_{CV}_fold_{database}{xtb_text}{s_text}.npy'
    save_file = f'slatm_baseline/slatm_hypers_{database}{xtb_text}{s_text}.csv'

    if not os.path.exists(slatm_save):
        maes_slatm = predict_CV(slatm, barriers, CV=CV, save_hypers=True,
                                    save_file=save_file)
        np.save(slatm_save, maes_slatm)
    else:
        maes_slatm = np.load(slatm_save)
    print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')
