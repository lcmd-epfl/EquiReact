import numpy as np
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial import distance_matrix
import pandas as pd
import os

def predict_KRR(X_train, X_test, y_train, y_test, gamma=0.001, l2reg=1e-10,
        kernel='laplacian'):
    """Perform KRR and return MAE of prediction.

    Args:
        X_train (np array): Training data
        X_test (np array): Data for out-of-sample prediction
        y_train (np array): Training labels
        y_test (np array): Labels for out-of-sample prediction
        l2reg (float): Regularizer. Default 1e-6
        gamma (float): gamma value for laplacian kernel. Default 1e-3
        kernel (str): whether kernel is rbf / gaussian, laplacian or polynomial

    Returns:
        float: Mean Absolute Error of prediction
    """

    K = laplacian_kernel(X_train, X_train, gamma=gamma)
    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)
    K_test = laplacian_kernel(X_test, X_train, gamma=gamma)

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred


def opt_hyperparams(
    X_train, X_val, y_train, y_val,
    gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    l2regs = [1e-10, 1e-7, 1e-4]
):
    """Optimize hyperparameters for KRR with
    laplacian kernel.
    """

    print("Hyperparam search for laplacian kernel")
     # Laplacian
    kernel = 'laplacian'
    maes_lap = np.zeros((len(gammas), len(l2regs)))
    for i, gamma in enumerate(gammas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = predict_KRR(
                X_train, X_val, y_train, y_val, gamma=gamma, l2reg=l2reg, kernel=kernel
                )
            print(f'mae={mae} for params {kernel, gamma, l2reg}')
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

    for i in range(CV):
        print("CV iteration", i)
        seed += i
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, random_state=seed, test_size=test_size)
        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, shuffle=False, test_size=0.5)
        print('train size', len(X_train), 'val size', len(X_val), 'test size', len(X_test))

        # hyperparam opt 

        if not os.path.exists(save_file):
            print("Optimising hypers...")
            gamma, l2reg = opt_hyperparams(X_train, X_val, y_train, y_val)
            gammas.append(gamma)
            l2regs.append(l2reg)
        else:
            gamma = gammas[i]
            l2reg = l2regs[i]
            print(f"Optimal params gamma={gamma} l2reg={l2reg}")


        print("Making prediction with optimal params...")
        mae, _ = predict_KRR(X_train, X_test,
                                y_train, y_test, 
                               l2reg=l2reg,
                                gamma=gamma)
        maes[i] = mae

    if not os.path.exists(save_file) and save_hypers:
        print(f'saving hypers to {save_file}')
        hypers = {"CV iter":np.arange(CV), "gamma":gammas, "l2reg":l2regs}
        df = pd.DataFrame(hypers)
        df.to_csv(save_file)
    return maes

