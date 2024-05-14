N = 3
left_gap = 0.02
right_gap = 0.2
x_gap = 0.02
x_plot = (1 - (left_gap+right_gap+x_gap*(N-1)))/N

M = 2
top_gap = 0.085
bottom_gap = 0.02
y_gap = 0.02
y_plot = (1 - (top_gap+bottom_gap+y_gap*(M-1)))/M

set size square

set term png size 1420,800

set output 'tsne.png'
unset xtics
unset ytics

set pm3d map
set xrange[-120:120]
set yrange[-120:120]

set multiplot layout M,N
INPS = '3dreact-gdb-inv-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat.tsne MPN.npy.tsne slatm_gdb.npy.tsne'
TITLES = '3DReact ChemProp SLATM_d'

set palette rgb 33,13,10  # rainbow
set cblabel '{/*1.2 Target (computed barrier), kcal/mol}' offset 1
set cbtics 50
set cbtics format "{/*1.2 %h}"

j=1
y1 = top_gap + (j-1)*(y_plot+y_gap)
y2 = y1 + y_plot
set tmargin screen 1-y1 ; set bmargin screen 1-y2

do for [i=1:N]{
  if (i==N){
    set colorbox
  } else{
    unset colorbox
  }

  x1 = left_gap + (i-1)*(x_plot+x_gap)
  x2 = x1 + x_plot
  set lmargin screen x1 ; set rmargin screen x2

  INP = word(INPS, i)
  set title '{/:Bold*1.5 '. word(TITLES, i) . '}'
  plot INP.'.targets.perp=64.0.ex=12.0.dat' u 1:2:3 with points pt 1 ps 1 palette noti

}

unset title
unset cbtics
#set cblabel 'Reaction class'
unset cblabel

set palette maxcolors 6
set cbrange [0:6]

set palette defined ( 0  'grey',      1 'grey',        \
                      1  '#8000ff',   1 '#8000ff',     \
                      2  '#1996f3',   2 '#1996f3',     \
                      3  '#b3f396',   4 '#b3f396',     \
                      4  '#ff964f',   5 '#ff964f',     \
                      5  '#ff0000',   6 '#ff0000'      \
                      )

set cbtics ('{/*1.2 Other}' 0, \
 "{/*1.2 +C–H,–C–C,–C–H}" 1, \
 "{/*1.2 +C–H,–C–H}" 2, \
 "{/*1.2 +H–H,–C–H,–C–H}" 3, \
 "{/*1.2 +H–N,–C–H}" 4, \
 "{/*1.2 +H–O,–C–H,–C–O}" 5) offset 0,1

j=2
y1 = top_gap + (j-1)*(y_plot+y_gap)
y2 = y1 + y_plot
set tmargin screen 1-y1 ; set bmargin screen 1-y2

do for [i=1:N]{
  if (i==N){
    set colorbox
  } else{
    unset colorbox
  }

  x1 = left_gap + (i-1)*(x_plot+x_gap)
  x2 = x1 + x_plot
  set lmargin screen x1 ; set rmargin screen x2

  INP = word(INPS, i)
  plot INP.'.bonds.perp=64.0.ex=12.0.dat' u 1:($3==0?$2:1/0):3 with points pt 7 ps .3 lc rgb "grey" noti,\
       ''                                 u 1:($3!=0?$2:1/0):3 with points pt 7 ps .3 palette noti
}
