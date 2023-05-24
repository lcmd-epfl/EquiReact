# TODO

## Cyclo
- Here the baseline is around 3kcal/mol for a non-ensembled model (equivalent)
- We can publish when we beat the baseline using atom mapping info (on par with their info), also ideally would like to beat baseline without atom mapping 

1. Opt params for each mode
    1. Energy: have tried/submitted variations of dropout, nconv, ns, nv, ngauss, radius / max neighbors, sum mode(node/edge/both), combine mode (diff/sum/mean/MLP). MLP is not working!
   2. Graph: have submitted with basic params, need to opt params and implement different modes (mean/sum but also maybe an MLP/nonlinear version)
   3. Atom mapping as a third option: not working as well as expected
2. Need to run baseline with atuomatic atom mapping for fairer comparison
3. When everything is ok we need to cross-validate numbers eg 5-fold 90-5-5 splits (these are the splits used in the baseline)

### Questions:
- [ ] radius, n neighbours
- [x] why MLP not working?
- [ ] can we use atom mapping in gdb smiles?
- [ ] how to use atom mapping info better in cyclo?

## GDB
- The baseline with atom mapping is around 3kcal/mol

1. Need to see how the atom mapped smiles can be used 
2. Need to run baseline with automatic atom mapping, which will be a fairer reference, since these reactions are complicated to atom map by hand
