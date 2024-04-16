set term pdf

set style fill solid 0.5 border -1
set style boxplot outliers pointtype 7
set style data boxplot
set boxwidth  0.5
set pointsize 0.1

models = "+C-H,-C-C,-C-H +C-H,-C-H +H-H,-C-H,-C-H +H-N,-C-H +H-O,-C-H,-C-O"

set output 'class-error-noH.pdf'
plot for [i=0:4] 'class-errors-true-noH.dat' using (i):1 index i ti word(models,i+1)
set output 'class-error-withH.pdf'
plot for [i=0:4] 'class-errors-true-withH.dat' using (i):1 index i ti word(models,i+1)
