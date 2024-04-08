import numpy as np
from process.dataloader_chemprop import get_scaffold_splits

def split_dataset(nreactions, splitter, tr_frac, dataset, subset=None):
    # seed before calling
    indices = np.arange(nreactions)
    len_before = len(indices)
    np.random.shuffle(indices)
    len_after = len(indices)
    assert len_before == len_after, "lost data in shuffle"
    if subset:
        indices = indices[:subset]
        assert len(indices) == subset, "lost data in subset"

    te_frac = (1. - tr_frac) / 2
    tr_size = round(tr_frac * len(indices))
    te_size = round(te_frac * len(indices))
    va_size = len(indices) - tr_size - te_size

    if splitter == 'random':
        print("Using random splits")
        tr_indices, te_indices, val_indices = np.split(indices, [tr_size, tr_size+te_size])

    elif splitter in ['yasc', 'ydesc']: # splits based on the target value
        print(f"Using target-based splits ({'ascending' if splitter=='yasc' else 'descending'} order)")
        idx4idx = np.argsort(np.array(data.labels[indices]))
        if splitter == 'ydesc':
            idx4idx = idx4idx[::-1]
        indices = indices[idx4idx]
        tr_indices, val_indices, te_indices = np.split(indices, [tr_size, tr_size+te_size])
        np.random.shuffle(tr_indices)
        np.random.shuffle(te_indices)
        np.random.shuffle(val_indices)

    elif splitter == 'scaffold':
        print("Using scaffold splits")
        tr_indices, te_indices, val_indices = get_scaffold_splits(dataset=dataset,
                                                                  indices=indices,
                                                                  sizes=(tr_frac, 1-(tr_frac+te_frac), te_frac))
    return tr_indices, te_indices, val_indices, indices

