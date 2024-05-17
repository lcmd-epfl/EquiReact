# Data

Each dataset ([`gdb`](gdb/) for GDB7-22-TS, [`cyclo`](cyclo/) for Cyclo-23-TS, [`proparg`](proparg/) for Proparg-21-TS) directory contains:
* `xyz/` — the original (DFT) geometries.
* `xyz-xtb/` — GFN2-xTB geometries.
* `{dataset}.csv` — the CSV file that contains:
  * `idx` / `rxn_id` / (`mol`,`enan`): reaction indices used to find the corresponding xyz files.
  * `dE0` / `G_act` / `Eafw`: target property.
  * `rxn_smiles`: unmapped reaction SMILES
  * `rxn_smiles_mapped`: the original ("true") atom-mapped SMILES
  * `rxn_smiles_rxnmapper`: SMILES mapped by RXNMapper
  * `rxn_smiles_rxnmapper_full`: SMILES mapped by RXNMapper including hydrogens
  * `bad_xtb`: is the reaction is excluded from the geometry quality tests (xTB optimization failed)

Additionally,
  * `cyclo/matches`: xyz atom-maps (see ../data-curation/cyclo-atom-mapping/).
  * `proparg/proparg-weird-smiles.csv`: "bad" SMILES for Proparg-21-TS automatically obtained from xyz
    taken from [doi:10.1039/d3dd00175j](https://doi.org/10.1039/d3dd00175j) ([repo](https://github.com/lcmd-epfl/benchmark-barrier-learning/)).
    They are also mapped by RXNMapper but were not used to produce the results of the paper. 
