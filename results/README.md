# Results

* [`get-results-from-wandb.py`](./get-results-from-wandb.py):
  downloads the results from [W&B](https://wandb.ai/equireact/projects), the output is saved in [`results.txt`](results.txt).
  * [`results.txt`](results.txt) contains 3DReact's results per run (run id, run name, seed, MAE, RMSE)
  and averaged across 10 folds (run name, mean MAE, MAE stdev, mean RMSE, RMSE stdev).

* [`get-tables.py`](get-tables.py) takes the errors from [`results.txt`](results.txt),
  [`../baseline_chemprop/results/*/fold_?/test_scores.csv`](../baseline_chemprop/results/),
  [`../baseline_slatm/results/slatm_10_fold_*.npy`](../baseline_slatm/results/)
  and produces the $\mathrm{\LaTeX}$ tables for the paper and data for gnuplot scripts. 
  * [`auto-generated-tables.dat`](auto-generated-tables.dat)
  * [`auto-generated-data-for-extrapolation-plot/`](auto-generated-data-for-extrapolation-plot)

* [`checkpoints/`](checkpoints): several checkpoint files of 3DReact used to generate the figures in the paper.
  * use [`../evaluation.py`](../evaluation.py) and [`../representation.py`](../representation.py)
    to compute the errors by reaction and learned representation, respectively.
  * [`log2dat.bash`](log2dat.bash): script to extract the errors by reaction from the output of `evaluation.py`. A similar script could be used
    to extract the learned representation fro the output of `representation.py`.
  * [`by_mol/`](by_mol/): the predictions by reaction: reaction index (in the dataset), target, prediction (in the dataset stdev units).

Scripts used to generate data for some of the Figures for the paper (see the READMEs inside): 
* [`repr/`](repr) — for the t-SNE maps.
* [`gdb-reaction-classes/`](gdb-reaction-classes) — for the error distributions per reaction class.
* [`cyclo-rmsd/`](cyclo-rmsd) — for the geometry sensitivity in the case of the Cyclo-23-TS set.
