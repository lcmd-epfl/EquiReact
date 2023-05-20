#!/usr/bin/env bash

N=5973
DDIR=graphs

for i in `seq 0 ${N}`; do
  if [ ! -f ${DDIR}/TS_${i}.dat ]; then
    continue
  fi
  ./22-assert-reactants.py ${DDIR}/{TS,R0,R1}_${i}.dat
done
