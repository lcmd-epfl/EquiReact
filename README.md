# Try to improve atom mapping / attention performance
- [x] run vector mode to compare
- [x] try random atom mapping
- [ ] play with atom_diff_nonlin
- [ ] try to add edge features to attention/mapping/vector mode

# Compete for GDB 
- [ ] Train on backwards and forward reactions
- [ ] Pass messages between reactants and well as between reactants & products

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
- The baseline with atom mapping is around 4.1kcal/mol

