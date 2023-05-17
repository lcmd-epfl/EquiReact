#!/usr/bin/env bash

N=5973
SDIR=../../data/cyclo/xyz
DDIR=graphs
mkdir -p ${DDIR}

for i in `seq 0 ${N}`; do

  RDIR=${SDIR}/${i}
  if [ ! -d ${RDIR} ]; then
    continue
  fi
  echo ${i}

  TS=${RDIR}/TS_*_*_*_*_*.xyz
  R0=${RDIR}/r0_*_alt.xyz
  R1=${RDIR}/r1_*_alt.xyz
  if ! compgen -G "$R0" > /dev/null; then
    R0=${RDIR}/r0_*.xyz
  fi
  if ! compgen -G "$R1" > /dev/null; then
    R1=${RDIR}/r1_*.xyz
  fi

  echo p | v ${TS} gui:0 > ${DDIR}/TS_${i}.dat
  echo p | v ${R0} gui:0 > ${DDIR}/R0_${i}.dat
  echo p | v ${R1} gui:0 > ${DDIR}/R1_${i}.dat

done
