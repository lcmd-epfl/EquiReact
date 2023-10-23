import numpy as np
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial import distance_matrix
import pandas as pd
import os
from process.dataloader_chemprop import get_scaffold_splits
from slatm_baseline.hypers import HYPERS

def predict_laplacian_KRR(D_train, D_test,
                y_train, y_test,
                gamma=0.001, l2reg=1e-10):
    """Perform KRR and return MAE of prediction using laplacian kernel.

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

def predict_gaussian_KRR(D_train, D_test,
                        y_train, y_test,
                        sigma=100, l2reg=1e-10):
    """
    Now for gaussian kernel
    """
    K      = np.exp(-D_train / 2*sigma**2)
    K_test = np.exp(-D_test / 2*sigma**2)
    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred

def opt_hyperparams_laplacian(
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
            mae, y_pred = predict_laplacian_KRR(
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

def opt_hyperparams_gaussian(
    D_train, D_val,
    y_train, y_val,
    sigmas = [1, 10, 100, 1e3, 1e4],
    l2regs = [1e-10, 1e-7, 1e-4]
):
    """Optimize hyperparameters for KRR with
    gaussian kernel.
    """

    print("Hyperparam search for gaussian kernel")
    maes = np.zeros((len(sigmas), len(l2regs)))

    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = predict_gaussian_KRR(
                D_train, D_val, y_train, y_val, sigma=sigma, l2reg=l2reg
                )
            print(f'{mae=} for params {sigma=} {l2reg=}')
            maes[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_i]
    min_l2reg = l2regs[min_j]
    min_mae = maes[min_i, min_j]

    print(f"Best mae={min_mae} for gamma={min_sigma} and l2reg={min_l2reg}")

    return min_sigma, min_l2reg


def predict_CV(X, y, CV=10, seed=1, train_size=0.8, kernel='laplacian',
               splitter='random', dataset=''):

    if kernel == 'laplacian':
        predict_KRR = predict_laplacian_KRR
        opt_hyperparams = opt_hyperparams_laplacian

        if dataset in HYPERS.keys():
            gamma, l2reg = HYPERS[dataset]

        D_full = pairwise_distances(X, metric='l1')

    elif kernel == 'rbf' or kernel == 'gaussian':
        predict_KRR = predict_gaussian_KRR
        opt_hyperparams = opt_hyperparams_gaussian

        if dataset in HYPERS.keys():
            sigma, l2reg = HYPERS[dataset]

        D_full = pairwise_distances(X, metric='l2')

    else:
        raise NotImplementedError("Only rbf/gaussian kernel or laplacian kernel are implemented.")

    maes = np.zeros((CV))

    for i in range(CV):
        print("CV iteration", i)
        seed += i

        if splitter == 'random':
            idx_train, idx_test_val = train_test_split(np.arange(len(y)), random_state=seed, train_size=train_size)
            idx_test, idx_val = train_test_split(idx_test_val, shuffle=False, test_size=0.5)
        elif splitter == 'scaffold':
            idx_train, idx_test, idx_val = get_scaffold_splits(dataset=dataset,
                                                                sizes=(train_size, (1-train_size)/2,
                                                                (1-train_size)/2))


        D_train = D_full[np.ix_(idx_train, idx_train)]
        D_val   = D_full[np.ix_(idx_val,   idx_train)]
        D_test  = D_full[np.ix_(idx_test,  idx_train)]
        y_train = y[idx_train]
        y_val   = y[idx_val]
        y_test  = y[idx_test]

        print('train size', len(y_train), 'val size', len(y_val), 'test size', len(y_test))
        if i == 0:
            # hyperparam opt
            if dataset not in HYPERS.keys():
                print("Optimising hypers...")
                param, l2reg = opt_hyperparams(D_train, D_val, y_train, y_val)

                if kernel == 'laplacian':
                    gamma = param
                elif kernel == 'rbf' or kernel == 'gaussian':
                    sigma = param

        if kernel == 'laplacian':
            print(f"Making prediction with optimal params gamma={gamma},l2reg={l2reg}")
            mae, _ = predict_KRR(D_train, D_test,
                                 y_train, y_test,
                                 l2reg=l2reg, gamma=gamma)
        elif kernel == 'rbf' or kernel == 'gaussian':
            print(f"Making prediction with optimal params sigma={sigma},l2reg={l2reg}")
            mae, _ = predict_KRR(D_train, D_test,
                                 y_train, y_test,
                                 l2reg=l2reg, sigma=sigma)

        maes[i] = mae

    return maes

