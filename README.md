# New Equireact

## Current status 
### Cyclo dataset
- after having found bug in SMILES vs xyz file, added stronger assert statements to check in dataloader / graph creation
- mod_dataset.csv contains the subset of the original data which passed SMILES conversion, but it seems this might depend on the version of rdkit, so I will later upload the .pt files somewhere (too big for github)
- Now trying to run the same tests as before on the new graphs 

#### #TODO on this set
- Change all params again 
- Check vector mode (need to update again for gpu, there are tensor type issues)

### GDB set 
- Ksenia is working on dataloader 

#### #TODO on this set 
- All param opt
- Both modes