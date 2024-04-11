import random
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from process.dataloader_chemprop import get_scaffold_splits
from baseline_slatm.hypers import HYPERS
from process.splitter import split_dataset

def opt_hyperparams_w_kernel(X, y, idx_train, idx_val, get_gamma,
                            sigmas, l2regs=[1e-10,1e-7,1e-4]):
    """Optimize hypers including laplacian/rbf kernel search"""
    D_full_laplace = compute_manhattan_dist(X)
    D_full_rbf = compute_euclidean_dist_squared(X)

    sigmas_opt = np.zeros(2)
    gammas_opt = np.zeros(2)
    l2regs_opt = np.zeros(2)
    maes_opt = np.zeros(2)
    for i, D_full in enumerate([D_full_rbf, D_full_laplace]):
        D_train = D_full[np.ix_(idx_train, idx_train)]
        D_val = D_full[np.ix_(idx_val, idx_train)]
        y_train = y[idx_train]
        y_val = y[idx_val]

        sigma, l2reg, mae = opt_hyperparams(D_train, D_val, y_train, y_val, get_gamma,
                                            sigmas=sigmas, l2regs=l2regs)
        sigmas_opt[i] = sigma
        gammas_opt[i] = get_gamma(sigma)
        l2regs_opt[i] = l2reg
        maes_opt[i] = mae
    if maes_opt[1] < maes_opt[0]:
        print(f"Best MAE {maes_opt[1]} for laplacian kernel, gamma={gammas_opt[1]} and l2reg={l2regs_opt[1]}")
        return 'laplacian', gammas_opt[1], l2regs_opt[1], D_full_laplace
    else:
        print(f"Best MAE {maes_opt[0]} for rbf kernel, gamma={gammas_opt[0]} and l2reg={l2regs_opt[0]}")
        return 'rbf', gammas_opt[0], l2regs_opt[0], D_full_rbf


def opt_hyperparams(D_train, D_val,
                    y_train, y_val, get_gamma,
                    sigmas, l2regs=[1e-10,1e-7,1e-4]):
    """Optimize hyperparameters for KRR

    Args:
        D_train (np array): Distance matrix for the training data
        D_test (np array): Distance matrix between training and out-of-sample
        y_train (np array): Training labels
        y_test (np array): Labels for out-of-sample prediction
        sigmas (np array): Kernel widths / inverse widths / etc in convenient units
        l2regs (np array): Regularizers
        get_gamma (func x): Function that converts sigma to gamma
                            so that kernel is computed as exp(-gamma * D)

    Returns:
        float: Mean Absolute Error of prediction
    """

    if len(sigmas) == 0 or len(l2regs) == 0:
        raise ValueError("Need to provide a list/array of sigma values")

    maes = np.zeros((len(sigmas), len(l2regs)))

    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred, _ = predict_KRR(
                D_train, D_val, y_train, y_val, gamma=get_gamma(sigma), l2reg=l2reg
                )
            maes[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_i]
    min_l2reg = l2regs[min_j]
    min_mae = maes[min_i, min_j]
    return min_sigma, min_l2reg, min_mae


def predict_KRR(D_train, D_test,
                y_train, y_test,
                gamma=0.001, l2reg=1e-10):
    """Perform KRR and return MAE of prediction using laplacian/gaussian kernel.

    Args:
        D_train (np array): Distance matrix for the training data
        D_test (np array): Distance matrix between training and out-of-sample
        y_train (np array): Training labels
        y_test (np array): Labels for out-of-sample prediction
        gamma (float): gamma value for the kernel. Default 1e-3
        l2reg (float): Regularizer. Default 1e-10

    Returns:
        float: Mean Absolute Error of prediction
    """
    # IMPORTANT: assuming D_train/D_test are computed appropriately
    # with l1/l2 according to gaussian or laplacian kernel
    # AND squared if necessary for rbf

    K      = np.exp(-gamma*D_train)
    K_test = np.exp(-gamma*D_test)

    K[np.diag_indices_from(K)] += l2reg
    alpha = np.linalg.solve(K, y_train)

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test-y_pred)**2))
    return mae, rmse, y_pred


def compute_manhattan_dist(X):
    try:  # use qstack C routine if running on ksenia's desktop SORRY
        print('try to use q-stack routine')
        import ctypes
        D_full = np.zeros((len(X), len(X)))
        array_2d_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')
        qstack_manh_path = '/home/xe/GIT/Q-stack/qstack/regression/lib/manh.so'
        qstack_manh = ctypes.cdll.LoadLibrary(qstack_manh_path).manh
        qstack_manh.restype = ctypes.c_int
        qstack_manh.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, array_2d_double, array_2d_double, array_2d_double]
        qstack_manh(len(X), len(X), len(X[0]), X, X, D_full)
    except:
        print('using default routine')
        D_full = pairwise_distances(X, metric='l1')
    return D_full


def compute_euclidean_dist_squared(X):
    D_full = pairwise_distances(X, metric='l2')
    D_full *= D_full
    return D_full


def predict_CV(X, y, CV=10, seed=123, train_size=0.8, kernel='laplacian',
               save_predictions=None,
               splitter='random', dataset=''):

    if kernel == 'laplacian':
        sigmas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        get_gamma = lambda x: x
    elif kernel == 'rbf' or kernel == 'gaussian':
        sigmas = [1, 10, 100, 1e3, 1e4]
        get_gamma = lambda x: 0.5 / x**2
    else:
        raise NotImplementedError("Only rbf/gaussian kernel or laplacian kernel are implemented.")
    l2regs = [1e-10, 1e-7, 1e-4]

    maes = np.zeros(CV)
    rmses = np.zeros(CV)

    if dataset in HYPERS.keys():
        print("Reading optimal hypers from file")
        kernel, gamma, l2reg = HYPERS[dataset]
        if kernel == 'laplacian':
            D_full = compute_manhattan_dist(X)
        elif kernel == 'rbf' or kernel == 'gaussian':
            D_full = compute_euclidean_dist_squared(X)

    for i in range(CV):
        print(f"CV iter {i+1}/{CV}")

        np.random.seed(seed)
        random.seed(seed)
        idx_train, idx_test, idx_val, _ = split_dataset(nreactions=len(y), splitter=splitter,
                                                        tr_frac=train_size, dataset=dataset)
        if i==0:
            print(f'train size {len(idx_train)} val size {len(idx_val)} test size {len(idx_test)}')
            if dataset not in HYPERS.keys():
                print("Optimizing hyperparameters")
                kernel, gamma, l2reg, D_full = opt_hyperparams_w_kernel(X, y, idx_train, idx_val, get_gamma,
                                                                        sigmas=sigmas, l2regs=l2regs)
        print(f"Making prediction with optimal params {kernel=}, {gamma=}, {l2reg=}")

        D_train = D_full[np.ix_(idx_train, idx_train)]
        D_test  = D_full[np.ix_(idx_test,  idx_train)]
        y_train = y[idx_train]
        y_test  = y[idx_test]

        mae, rmse, y_pred = predict_KRR(D_train, D_test, y_train, y_test, l2reg=l2reg, gamma=gamma)

        with open(save_predictions.format(i=i), 'w') as f:
            print(*zip(idx_test, y_pred), sep='\n', file=f)

        maes[i] = mae
        rmses[i] = rmse
        seed += 1

    return maes, rmses
