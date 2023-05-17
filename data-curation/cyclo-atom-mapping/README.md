1. Create molecular graphs
```
./11-get-graphs.bash
```
# Create molecular graphs based on atom distance using
# https://github.com/briling/v/tree/55ba42f9ba56a3c37feb7f9651d0940b48239159

2.1. Check if the TS graphs have exactly 2 connected components:
```
./21-assert-components.bash
```
# Then the TSs where the assertion for the number of connected components failed
# (   846 3090 3524 3526 3701
#    3710 3766 3923 3926 4073
#    4125 4216 4220 4295 4457
#    4584 4594 4678 5201 5765   )
# were fixed manually.
#
# TS 3090: remove bonds 3-23 4-24
# TS 3710: remove bond 9-13
# TS 3766: remove bond 4-18
# TS 4295: remove bond 5-6
# other: just increase the cutoff

2.2. Check if the components have the same composition and number of bonds as the reactants
and there is only one way to find which reactant is which component
```
./22-assert-reactants.bash
```
# Then the TSs where the assertion for the components composition failed
# were fixed manually as well:
# TS 3065 add bond 4-14
# TS 4069 add bond 3-4
# TS 4727 add bond 10-11
# TS 5930 remove bond 10-20

3. Generate the atom mappings
```
./31-get-matches.bash
```
