set lt 1 lc rgb 0x11a8ca
set lt 2 lc rgb 0xff9c00
set lt 3 lc rgb 0x48b62d
PS=2

set size square
set xtics scale 0.5 1 offset 0,0.25
set ytics scale 0.5 4
set mxtics 2
set mytics 2
set ylabel "abs. error (xtb) – abs. error (DFT), kcal/mol" offset 1
set xlabel "xtb–DFT RMSD, Å" offset 0,0.5
set key samplen 0.5

set title font ",20"

set xrange [1:6]
set yrange [-12:12]

plot 0 ls 0 noti, \
     'cyclo_3dreact_forplot.dat' pt  9 ps PS ti '3DReact',\
     'cyclo_slatm_forplot.dat'   pt 11 ps PS ti 'SLATM_d'
