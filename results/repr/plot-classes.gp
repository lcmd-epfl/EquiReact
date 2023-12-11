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

set palette defined ( 0  "grey",      1 "grey",     \
                      1  "blue",      2 "blue",     \
                      2  "red",       3 "red",       \
                      3  "orange" ,   4 "orange" ,   \
                      4  "green",     5 "green",     \
                      5  "yellow",    6 "yellow",    \
                      6  "magenta",         7  "magenta",        \
                      7  "cyan",            8  "cyan",           \
                      8  "dark-violet",     9  "dark-violet",    \
                      9  "black",           10 "black",          \
                      10 "dark-yellow",     11 "dark-yellow",    \
                      11 "forest-green",    12 "forest-green",   \
                      12 "brown",           13 "brown",          \
                      13 "navy",            14 "navy",           \
                      14 "plum"                       \
                      )


plot INP u 1:($3==0?$2:1/0):3 with points pt 7 ps .3 lc rgb "grey" noti,\
     ''  u 1:($3!=0?$2:1/0):3 with points pt 7 ps .3 palette noti
