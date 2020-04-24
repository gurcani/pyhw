#include <fftw3.h>
#include <complex.h>

typedef struct fhwak_pars_{
  fftw_plan plan6f,plan6b,plan2f,plan2b;
  complex *dat,*linmat,*fk_phi,*fk_n;
  int nx,ny,npadx,npady;
  double *kx,*ky;
}fhwak_pars;

void fhwak_init(int nx,int ny,int npadx,int npady,complex *dat, complex *linmat, double *kx, double *ky,complex *fk_phi,complex *fk_n);
void fftw_hwak(complex *y, complex *dydt);
void fhw_pad(complex *in,complex *out);
