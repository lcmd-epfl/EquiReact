# Code to obtain the ChemProp results

We use [`chemprop` version 1.5.0](https://github.com/chemprop/chemprop/tree/v1.5.0)

* [`src/cgr.py`](src/cgr.py) is the main script used to train the model for the specified dataset, atom-mapping regime, and split type.
It can be run, for example, for the GDB7-22-TS using the True atom-mapping, random splits, and implicit H nodes with
```bash
python src/cgr.py --gdb --true
```
Note that the training creates a lot of directories and output files so it is more convenient to run the script from a dedicated directory.
* [`src/cgr-repr.py`](src/cgr-repr.py) extracts and saves a representation for a given model checkpoint. See example of usage at [`../results/repr`](../results/repr).
* [`src/chemprop.patch`](src/chemprop.patch) is the patch we used to bypass the valence check (see below).
* [`config/`](config) contains the hyperparameters for each dataset taken from [doi:10.1039/d3dd00175j](https://doi.org/10.1039/d3dd00175j) ([repo](https://github.com/lcmd-epfl/benchmark-barrier-learning/)).
* [`results/`](results) contains the submission scripts and the results (MAEs and RMSEs) for all the models trained, as well as
the checkpoint file used to generate the t-SNE map ([`results/gdb-true/fold_0/fold_0/model_0/model.pt`](results/gdb-true/fold_0/fold_0/model_0/model.pt),
see [`../results/repr`](../results/repr)).

#### Patch
The original `chemprop` code cannot process hypervalent Si compounds of the Proparg-21-TS dataset.
The patch that disables the valence check can be found at [`src/chemprop.patch`](src/chemprop.patch)
Run
```bash
$ patch < src/chemprop.patch
```
to apply it and
```bash
$ patch -R < src/chemprop.patch
```
to revert. (Modification of the paths may be needed.)

