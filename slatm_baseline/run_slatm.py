import argparse as ap
from slatm_baseline.reaction_reps import QML
from slatm_baseline.learning import predict_CV
import numpy as np
import os

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--database', default='gdb')
    args = parser.parse_args()
    database = args.database
    print("Running for database", database)

    qml = QML()

    if database == 'gdb':
        qml.get_GDB7_ccsd_data()
        slatm_save = 'slatm_baseline/slatm_gdb.npy'

    elif database == 'cyclo':
        qml.get_cyclo_data()
        slatm_save = 'slatm_baseline/slatm_cyclo.npy'

    barriers = qml.barriers

    CV = 10
    if not os.path.exists(slatm_save):
        slatm = qml.get_SLATM()
        np.save(slatm_save, slatm)
    else:
        slatm = np.load(slatm_save)

    if database == 'gdb':
        slatm_save = f'slatm_baseline/slatm_{CV}_fold_gdb.npy'
        save_file = 'slatm_baseline/slatm_hypers_gdb.csv'
    elif database == 'cyclo':
        slatm_save = f'slatm_baseline/slatm_{CV}_fold_cyclo.npy'
        save_file = 'slatm_baseline/slatm_hypers_cyclo.csv'

    if not os.path.exists(slatm_save):
        maes_slatm = predict_CV(slatm, barriers, CV=CV, save_hypers=True,
                                    save_file=save_file)
        np.save(slatm_save, maes_slatm)
    else:
        maes_slatm = np.load(slatm_save)
    print(f'slatm mae {np.mean(maes_slatm)} +- {np.std(maes_slatm)}')
