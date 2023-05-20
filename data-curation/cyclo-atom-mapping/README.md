# Code to generate atom mapping for the cyclo dataset

The procedure is based on the following assumptions:
1) atom numbering is the same for the transition state structures and the product structures;
2) (most of) the transition states are early, i.e. consist of two deformed reactants that can be clearly separated.

In principle, many atom mappings are possible (e.g. up to permutation of Hydrogens in a CH3 group),
and we simply choose the first one from the list generated.
Thus, enantiotopic and diastereotopic atoms could be switched.


## Requirements:
- [`networkx==2.8`](https://github.com/networkx/networkx) for graph manipulation
- [`https://github.com/briling/v`](https://github.com/briling/v) for creation of molecular graphs (can be substituted by any analog)

## How to reproduce:

### 1. Create the molecular graphs
```
./11-get-graphs.bash
```
Create the molecular graphs based on interatomic distances using
https://github.com/briling/v/tree/55ba42f9ba56a3c37feb7f9651d0940b48239159
and save them to `./graphs/`.

### 2.1. Check if the TS graphs have exactly 2 connected components
```
./21-assert-components.bash
```
Then fix the TSs for which the assertion has failed:
```
  846 3090 3524 3526 3701 3710 3766 3923 3926 4073
 4125 4216 4220 4295 4457 4584 4594 4678 5201 5765
```
Most of the structures one can just open in `v` (visual mode), decrease the bond cutoff, and save the graph.
For four structures a manual fix is needed:
```
TS 3090: remove bonds 3-23 4-24
TS 3710: remove bond 9-13
TS 3766: remove bond 4-18
TS 4295: remove bond 5-6
```

### 2.2. Check if the TS components have the same atom composition and number of bonds as the reactants
```
./22-assert-reactants.bash
```
Then manually fix the TSs for which the assertion has failed:
```
TS 3065: add bond 4-14
TS 4069: add bond 3-4
TS 4727: add bond 10-11
TS 5930: remove bond 10-20
```

### 3. Generate the atom mappings
```
./31-get-matches.bash
```
Generate one of the possible atom mappings and save to `./matches/`.
For each reaction `$i`, the files `./matches/R{0,1}_$i.dat` contain lists of integers indicating the position 
of the corresponding reactant atom in the product.
