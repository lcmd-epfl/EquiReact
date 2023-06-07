# Try to improve atom mapping / attention performance
- [x] run vector mode to compare
- [x] try random atom mapping
- [ ] play with atom_diff_nonlin
- [ ] try to add edge features to attention/mapping/vector mode

# Model parameters combinations

- `--graph_mode vector`
- `--graph_mode energy --combine_mode {diff,sum,mean,mlp} --sum_mode {node,edge,both}`
- `--attention {cross,self} --graph_mode {energy,vector} --combine_mode {diff,sum,mean,mlp}`
- `--atom_mapping {cross,self} --graph_mode {energy,vector} --combine_mode {diff,sum,mean,mlp}`

# TODO

## Cyclo
- Here the baseline is around 3kcal/mol
- We can publish when we beat the baseline using atom mapping info (on par with their info), also ideally would like to beat baseline without atom mapping 

- Opt params for each mode
    1. Energy: have tried/submitted variations of dropout, nconv, ns, nv, ngauss, radius / max neighbors, sum mode(node/edge/both), combine mode (diff/sum/mean/MLP). MLP is not working!
   2. Graph: have submitted with basic params, need to opt params and implement different modes (mean/sum but also maybe an MLP/nonlinear version)
   3. Atom mapping as a third option: not working as well as expected

### Questions:
- [ ] radius, n neighbours
- [x] why MLP not working?
- [x] can we add something nonlinear to MLP? 
- [x] can we use atom mapping in gdb smiles?
- [x] how to use atom mapping info better in cyclo?
- [x] learning rate? (why some LC oscillate so much)

## GDB
- The baseline with atom mapping is around 4.5kcal/mol

