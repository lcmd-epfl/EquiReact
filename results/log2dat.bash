#!/usr/bin/env bash

for LOG in by_mol/*.log; do
  ARGS=$(grep --max-count=1 wandb_name ${LOG} | tr -d \')
  SEED=${ARGS##*seed=}
  SEED=${SEED%%,*}
  NAME=${ARGS##*wandb_name=}
  NAME=${NAME%%,*}
  echo $SEED $NAME

  sed -n -e "/>>>/s/>>>//p" ${LOG} > ${NAME}.${SEED}.dat
done
