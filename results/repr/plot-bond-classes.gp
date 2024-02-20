#!/usr/bin/env -S gnuplot -c

INP=ARG1

set term svg size 400,400
set output INP.'.svg'
unset xtics
unset ytics

set pm3d map
set xrange[-120:120]
set yrange[-120:120]

set palette maxcolors 6
set cbrange [0:6]

#set palette rgb 33,13,10

#8000ff
#1996f3
#4df3ce
#b3f396
#ff964f
#ff0000

set palette defined ( 0  'grey',      1 'grey',        \
                      1  '#8000ff',   1 '#8000ff',     \
                      2  '#1996f3',   2 '#1996f3',     \
                      3  '#b3f396',   4 '#b3f396',     \
                      4  '#ff964f',   5 '#ff964f',     \
                      5  '#ff0000',   6 '#ff0000'      \
                      )

#set palette defined ( 0  'grey',      1 'grey',      \
#                      1  '#648fff',   2 '#648fff',   \
#                      2  '#785ef0',   3 '#785ef0',   \
#                      3  '#dc267f',   4 '#dc267f',   \
#                      4  '#fe6100',   5 '#fe6100',   \
#                      5  '#ffb000',   6 '#ffb000'    \
#                      )


plot INP u 1:($3==0?$2:1/0):3 with points pt 7 ps .3 lc rgb "grey" noti,\
     ''  u 1:($3!=0?$2:1/0):3 with points pt 7 ps .3 palette noti
