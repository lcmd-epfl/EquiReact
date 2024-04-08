import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from process.dataloader_chemprop import get_scaffold_splits
from slatm_baseline.hypers import HYPERS


def opt_hyperparams(D_train, D_val,
                    y_train, y_val,
                    sigmas, l2regs, get_gamma):
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

    maes = np.zeros((len(sigmas), len(l2regs)))

    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = predict_KRR(
                D_train, D_val, y_train, y_val, gamma=get_gamma(sigma), l2reg=l2reg
                )
            print(f'{mae=} for params {sigma=} {l2reg=}')
            maes[i, j] = mae
    min_i, min_j = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_i]
    min_l2reg = l2regs[min_j]
    min_mae = maes[min_i, min_j]
    print(f"Best mae={min_mae} for sigma={min_sigma} and l2reg={min_l2reg}")
    return min_sigma, min_l2reg


def predict_KRR(D_train, D_test,
                y_train, y_test,
                gamma, l2reg=1e-10):
    """Perform KRR and return MAE of prediction using laplacian/gaussian kernel.

    Args:
        D_train (np array): Distance matrix for the training data
        D_test (np array): Distance matrix between training and out-of-sample
        y_train (np array): Training labels
        y_test (np array): Labels for out-of-sample prediction
        gamma (float): gamma value for the kernel
        l2reg (float): Regularizer. Default 1e-10

    Returns:
        float: Mean Absolute Error of prediction
    """
    K      = np.exp(-gamma*D_train)
    K_test = np.exp(-gamma*D_test)

    K[np.diag_indices_from(K)] += l2reg
    alpha = np.linalg.solve(K, y_train)

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred


def compute_manhattan_dist(X):
    try:  # use qstack C routine if running on ksenia's desktop SORRY
        import ctypes
        D_full = np.zeros((len(X), len(X)))
        array_2d_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')
        qstack_manh_path = '/home/xe/GIT/Q-stack/qstack/regression/lib/manh.so'
        qstack_manh = ctypes.cdll.LoadLibrary(qstack_manh_path).manh
        qstack_manh.restype = ctypes.c_int
        qstack_manh.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, array_2d_double, array_2d_double, array_2d_double]
        qstack_manh(len(X), len(X), len(X[0]), X, X, D_full)
    except:
        D_full = pairwise_distances(X, metric='l1')
    return D_full


def compute_euclidean_dist_squared(X):
    D_full = pairwise_distances(X, metric='l2')
    D_full *= D_full
    return D_full


def predict_CV(X, y, CV=10, seed=1, train_size=0.8, kernel='laplacian',
               save_predictions=None,
               splitter='random', dataset=''):

    if kernel == 'laplacian':
        sigmas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        get_gamma = lambda x: x
        D_full = compute_manhattan_dist(X)
    elif kernel == 'rbf' or kernel == 'gaussian':
        sigmas = [1, 10, 100, 1e3, 1e4]
        get_gamma = lambda x: 0.5 / x**2
        D_full = compute_euclidean_dist_squared(X)
    else:
        raise NotImplementedError("Only rbf/gaussian kernel or laplacian kernel are implemented.")
    l2regs = [1e-10, 1e-7, 1e-4]

    maes = np.zeros((CV))

    for i in range(CV):
        print("CV iteration", i)
        seed += i

        if splitter == 'random':
            idx_train, idx_test_val = train_test_split(np.arange(len(y)), random_state=seed, train_size=train_size)
            idx_test, idx_val = train_test_split(idx_test_val, shuffle=False, test_size=0.5)
        elif splitter == 'scaffold':
            indices = np.arange(len(y))
            np.random.shuffle(indices)
            idx_train, idx_test, idx_val = get_scaffold_splits(dataset=dataset,
                                                               indices=indices,
                                                               sizes=(train_size, (1-train_size)/2,
                                                               (1-train_size)/2))

        D_train = D_full[np.ix_(idx_train, idx_train)]
        D_val   = D_full[np.ix_(idx_val,   idx_train)]
        D_test  = D_full[np.ix_(idx_test,  idx_train)]
        y_train = y[idx_train]
        y_val   = y[idx_val]
        y_test  = y[idx_test]

        print(f'train size {len(y_train)} val size {len(y_val)} test size {len(y_test)}')
        if i==0:
            if dataset in HYPERS.keys():
                sigma, l2reg = HYPERS[dataset]
            else:
                print("Optimizing hyperparameters")
                sigma, l2reg = opt_hyperparams(D_train, D_val, y_train, y_val, sigmas=sigmas, l2regs=l2regs, get_gamma=get_gamma)
            gamma = get_gamma(sigma)

        print(f"Making prediction with optimal params {sigma=}, {l2reg=}")
        mae, y_pred = predict_KRR(D_train, D_test, y_train, y_test, l2reg=l2reg, gamma=gamma)

        with open(save_predictions.format(i=i), 'w') as f:
            print(*zip(idx_test, y_pred), sep='\n', file=f)

        maes[i] = mae

    return maes
