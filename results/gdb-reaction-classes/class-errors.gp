set term png size 1000,500
set output 'class-errors.png'

set size square
set style fill solid 0.5 border -1
set style boxplot outliers pointtype 13
set style data boxplot
set boxwidth  0.5
set pointsize 0.25
set style boxplot fraction 0.90

classes = "+C–H,–C–C,–C–H +C–H,–C–H +H–H,–C–H,–C–H +H–N,–C–H +H–O,–C–H,–C–O"

set linetype 1 lc rgb  '#8000ff'
set linetype 2 lc rgb  '#1996f3'
set linetype 3 lc rgb  '#b3f396'
set linetype 4 lc rgb  '#ff964f'
set linetype 5 lc rgb  '#ff0000'

set ylabel 'Error, stdev units' offset 1.5
unset xtics
set ytics 1 format "%.1f"
set yrange [-2.75:]
set key bottom left reverse Left
set mytics 2

set multiplot layout 1,2
set title 'with explicit H'
plot for [i=0:4] 'class-errors-cv10-gdb-inv-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat' using (i):1 index i ti word(classes, i+1)
set title 'without explicit H'
plot for [i=0:4] 'class-errors-cv10-gdb-inv-random-withH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat' using (i):1 index i ti word(classes, i+1)
