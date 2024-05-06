# Script to generate t-SNE maps for the GDB7-22-TS set

```bash
for REPR in MPN.npy slatm_gdb.npy 3dreact-gdb-inv-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat; do
    for COLOR in bonds targets; do
        ./dim-reduction.gdb.equireact.py --method tsne --dump --repr_path ${REPR} \
                                         --tsne_perp 64 --tsne_ex 12 --how_to_color ${COLOR};
    done
done
```

## How to get the representations:
### SLATM$_d$ representation
`../../baseline_slatm/slatm_gdb.npy`
is computed with the tools of [`/baseline_slatm/`](../../baseline_slatm/) (not in the repo because it's too big)

### 3DReact representation
`3dreact-gdb-inv-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat`
is obtained with
```bash
(cd ../../;
python representation.py --checkpoint results/checkpoints/cv10-gdb-inv-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both-*.log;
sed -n -e "/>>>/s/>>>//p" `ls logs/evaluation/*.log -rt | tail -n1` > results/repr/3dreact-gdb-inv-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat
)
```

### ChemProp representation:
`MPN.npy` is obtained with
```bash
python ../../baseline_chemprop/src/cgr-repr.py --fingerprint_type MPN \
       --checkpoint ../../baseline_chemprop/results/gdb-true/fold_0/fold_0/model_0/model.pt \
       --data_path ../../baseline_chemprop/csv/gdb.csv
```


