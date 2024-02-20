#!/usr/bin/env -S gnuplot -c

INP=ARG1

set term svg size 400,400
set output INP.'.svg'
unset xtics
unset ytics

set pm3d map
set xrange[-120:120]
set yrange[-120:120]

set palette rgb 33,13,10  # rainbow

plot INP u 1:2:3 with points pt 1 ps 1 palette noti
