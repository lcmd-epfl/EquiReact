#include <stdlib.h>
#include <math.h>

int manh(int I, int J, int K, double X[I][K], double Y[J][K], double D[I][J]){
  int same = (X==Y) && (I==J);
  #pragma omp parallel shared(D)
	#pragma omp for schedule(dynamic)
  for(int i=0; i<I; i++){
    int j0 = ( same ? i : 0 );
    for(int j=j0; j<J; j++){
      double d = 0;
      for(int k=0; k<K; k++){
        d += fabs(X[i][k]-Y[j][k]);
      }
      D[i][j] = d;
      if(same && i!=j){
        D[j][i] = d;
      }
   }
  }
  return 0;
}

int eucl(int I, int J, int K, double X[I][K], double Y[J][K], double D[I][J]){
  int same = (X==Y) && (I==J);
  #pragma omp parallel shared(D)
	#pragma omp for schedule(dynamic)
  for(int i=0; i<I; i++){
    int j0 = ( same ? i : 0 );
    for(int j=j0; j<J; j++){
      double d = 0;
      for(int k=0; k<K; k++){
        double x = X[i][k]-Y[j][k];
        d += x*x;
      }
      D[i][j] = d;
      if(same && i!=j){
        D[j][i] = d;
      }
   }
  }
  return 0;
}
