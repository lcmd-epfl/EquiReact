import numpy as np
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial import distance_matrix
import pandas as pd
import os

def predict_KRR(D_train, D_test,
                y_train, y_test,
                gamma=0.001, l2reg=1e-10):
    """Perform KRR and return MAE of prediction.

    Args:
        D_train (np array): Distance matrix for the training data
        D_test (np array): Distance matrix between training and out-of-sample
        y_train (np array): Training labels
        y_test (np array): Labels for out-of-sample prediction
        l2reg (float): Regularizer. Default 1e-10
        gamma (float): gamma value for laplacian kernel. Default 1e-3

    Returns:
        float: Mean Absolute Error of prediction
    """
    K      = np.exp(-gamma*D_train)
    K_test = np.exp(-gamma*D_test)

    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred


def opt_hyperparams(
    D_train, D_val,
    y_train, y_val,
    gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    l2regs = [1e-10, 1e-7, 1e-4]
):
    """Optimize hyperparameters for KRR with
    laplacian kernel.
    """

    print("Hyperparam search for laplacian kernel")
     # Laplacian
    maes_lap = np.zeros((len(gammas), len(l2regs)))

    for i, gamma in enumerate(gammas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = predict_KRR(
                D_train, D_val, y_train, y_val, gamma=gamma, l2reg=l2reg
                )
            print(f'{mae=} for params {gamma=} {l2reg=}')
            maes_lap[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes_lap, axis=None), maes_lap.shape)
    min_gamma = gammas[min_i]
    min_l2reg = l2regs[min_j]
    min_mae_lap = maes_lap[min_i, min_j]

    print(f"Best mae={min_mae_lap} for gamma={min_gamma} and l2reg={min_l2reg}")

    return min_gamma, min_l2reg


def predict_CV(X, y, CV=10, seed=1, test_size=0.2, save_hypers=False, save_file=''):

    maes = np.zeros((CV))
    if not os.path.exists(save_file):
        gammas= []
        l2regs = []
    else:
        hypers = pd.read_csv(save_file)
        gammas = hypers['gamma'].to_list()
        l2regs = hypers['l2reg'].to_list()


    D_full = pairwise_distances(X, metric='l1')

    for i in range(CV):
        print("CV iteration", i)
        seed += i

        idx_train, idx_test_val = train_test_split(np.arange(len(y)), random_state=seed, test_size=test_size)
        idx_test, idx_val = train_test_split(idx_test_val, shuffle=False, test_size=0.5)

        D_train = D_full[np.ix_(idx_train, idx_train)]
        D_val   = D_full[np.ix_(idx_val,   idx_train)]
        D_test  = D_full[np.ix_(idx_test,  idx_train)]
        y_train = y[idx_train]
        y_val   = y[idx_val]
        y_test  = y[idx_test]

        print('train size', len(y_train), 'val size', len(y_val), 'test size', len(y_test))

        # hyperparam opt

        if not os.path.exists(save_file):
            print("Optimising hypers...")
            gamma, l2reg = opt_hyperparams(D_train, D_val, y_train, y_val)
            gammas.append(gamma)
            l2regs.append(l2reg)
        else:
            gamma = gammas[i]
            l2reg = l2regs[i]
            print(f"Optimal params gamma={gamma} l2reg={l2reg}")


        print("Making prediction with optimal params...")
        mae, _ = predict_KRR(D_train, D_test,
                             y_train, y_test,
                             l2reg=l2reg, gamma=gamma)
        maes[i] = mae

    if not os.path.exists(save_file) and save_hypers:
        print(f'saving hypers to {save_file}')
        hypers = {"CV iter":np.arange(CV), "gamma":gammas, "l2reg":l2regs}
        df = pd.DataFrame(hypers)
        df.to_csv(save_file)
    return maes

