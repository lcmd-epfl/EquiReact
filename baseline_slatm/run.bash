#!/usr/bin/bash

for dataset in gdb proparg cyclo; do
    ./run_slatm.py -d ${dataset} --splitter random
    ./run_slatm.py -d ${dataset} --splitter random --xtb
    ./run_slatm.py -d ${dataset} --splitter random --no_h_atoms
    ./run_slatm.py -d ${dataset} --splitter random --no_h_total

    for splitter in scaffold sizeasc sizedesc yasc ydesc; do
        ./run_slatm.py -d ${dataset} --splitter ${splitter};
    done

    if [[ "$dataset" != "proparg" ]]; then
        ./run_slatm.py -d ${dataset} --splitter random --xtb_subset
    fi
done
