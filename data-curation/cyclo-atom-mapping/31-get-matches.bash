#!/usr/bin/env bash

N=5973
DDIR=graphs
PDIR=matches

for i in `seq 0 ${N}`; do
  if [ ! -f ${DDIR}/TS_${i}.dat ]; then
    continue
  fi
  #echo $i >> /dev/stderr
  ./31-get-matches.py ${DDIR}/{TS,R0,R1}_${i}.dat ${PDIR}/R{0,1}_$i.dat
done
