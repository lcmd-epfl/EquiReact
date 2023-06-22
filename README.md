# Try to improve atom mapping / attention performance
- [x] run vector mode to compare
- [x] try random atom mapping
- [ ] play with atom_diff_nonlin
- [ ] try to add edge features to attention/mapping/vector mode

# Compete for GDB 
- [ ] Train on backwards and forward reactions
- [ ] Pass messages between reactants and well as between reactants & products
- [ ] Add reaction enthalpy to features for prediction
- [ ]  Adding ring size to both the atom and bond features further improvesthe model.
- [ ]  See here : https://github.com/kspieks/chemprop/tree/barrier_prediction for changes
- [ ]  Longer term : rather than reactants and products, use some kind of TS guess and use TS too eg https://pubs.rsc.org/en/content/articlehtml/2020/cp/d0cp04670a

# Cyclo idea
- [ ] pre-train on GDB then fine-tune on cyclo / proparg ? other datasets

# Model parameters combinations

- `--graph_mode vector`
- `--graph_mode energy --combine_mode {diff,sum,mean,mlp} --sum_mode {node,edge,both}`
- `--attention {cross,self} --graph_mode {energy,vector} --combine_mode {diff,sum,mean,mlp}`
- `--atom_mapping True --graph_mode {energy,vector} --combine_mode {diff,sum,mean,mlp}`

# TODO

## Cyclo
- Here the baseline is around 3kcal/mol

### Questions:
- [ ] radius, n neighbours
- [x] why MLP not working?
- [x] can we add something nonlinear to MLP? 
- [x] can we use atom mapping in gdb smiles?
- [x] how to use atom mapping info better in cyclo?
- [x] learning rate? (why some LC oscillate so much)

## GDB
- The baseline with atom mapping is 4.9 without rxn enthalpy and without reverse/fwd reactions (my run)
- Reported baseline is 4.1 with rxn enthalpy and reverse/fwd reactions, 85/5/10 scaffold splits. 

