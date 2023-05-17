#!/usr/bin/env bash

N=5973
DDIR=graphs

for i in `seq 0 ${N}`; do
  if [ ! -f ${DDIR}/TS_${i}.dat ]; then
    continue
  fi
  echo $i >> /dev/stderr
  ./21-assert-components.py ${DDIR}/TS_${i}.dat
done
