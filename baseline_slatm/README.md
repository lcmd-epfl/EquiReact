# Code to obtain the $\mathrm{SLATM}_d$ results

* [`run_slatm.py`](run_slatm.py) — the main script that computes the representation (if not in `repr/`),
  optimizes the hyperparameters (if not in [`hypers.py`](hypers.py)), and computes the prediction errors (if not in [`results/`](results/))
* [`environment.yml`](environment.yml) — conda environment file
* [`run.bash`](run.bash) — wrapper script to recompute all the results
* [`hypers.py`](hypers.py) — models hyperparameters
* [`manh.c`](manh.c), [`makefile`](makefile) — optional C module for faster computation of Laplacian kernel
* [`learning.py`](), [`reaction_reps.py`]() — functions to perform KRR and compute $\mathrm{SLATM}_d$, respectively  
#### Produced results:
* `repr/` — computed representations (not in the repo because of the size)
* [`results/`](results/) — saved MAE/RMSD
* [`by_mol/`](by_mol/) — predictions for each test molecule
