# Code to atom-map the datasets with RXNMapper

We use [`rxnmapper` version 0.3.0](https://github.com/rxn4chemistry/rxnmapper/releases/tag/v0.3.0)

* [`mapper.py`](mapper.py) the wrapper over `RXNMapper`.
  Creates `gdb.csv` and `cyclo.csv` files in the current directory. 
* [`environment.yml`](environment.yml) — conda environment file.
* [`rxnmapper.patch`](rxnmapper.patch) — the patch we used to bypass the molecule sorting (see below).
* [`confidence.dat`](confidence.dat) mean confidence for different regimes and datasets.
  * [`cyclo-noH-confidence.dat`](cyclo-noH-confidence.dat) confidence for each reaction (Cyclo-23-TS).
  * [`gdb-noH-confidence.dat`](gdb-noH-confidence.dat) confidence for each reaction (GDB7-22-TS).

#### Patch
The original `RXNMapper` code sorts the reactant/products molecules (if several in a reaction). 
The patch that disables the sorting can be found at [`src/rxnmapper.patch`](src/rxnmapper.patch)
Run
```bash
$ patch < src/rxnmapper.patch
```
to apply it and
```bash
$ patch -R < src/rxnmapper.patch
```
to revert. (Modification of the paths may be needed.)

