#!/usr/bin/env bash

SDIR=../../data/gdb7-22-ts/xyz/
#DDIR=wrong-number-of-bonds/
DDIR=no-isomorphism/
mkdir -p ${DDIR}

for i in `ls $DDIR/*.png`; do
  MDIR=$(basename $i)
  MDIR=${MDIR:1:6}
  for XYZ in ${SDIR}/${MDIR}/p*.xyz; do
    echo p | v ${XYZ} gui:0 rot:0.528,0,0,0,0.528,0,0,0,0.528 > ${DDIR}/$(basename ${XYZ/.xyz/.dat})
    cp ${XYZ} ${DDIR}
  done
done
