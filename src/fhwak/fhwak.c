#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fhwak.h"
#include <fftw3.h>

fhwak_pars *fhwp;
#define NTH 8

void fhwak_init(int nx,int ny,int npadx,int npady,complex *dat, complex *linmat, double *kx, double *ky, complex *fk_phi, complex *fk_n){
  int  aux[2],N,Nc;
  fhwp=malloc(sizeof(fhwak_pars));
  fhwp->nx=nx;
  fhwp->ny=ny;
  fhwp->npadx=npadx;
  fhwp->npady=npady;
  fhwp->dat=dat;
  fhwp->kx=kx;
  fhwp->ky=ky;
  fhwp->fk_phi=fk_phi;
  fhwp->fk_n=fk_n;
  fhwp->linmat=linmat;  
  aux[0]=nx+2*npadx;
  aux[1]=ny+2*npady;
  N=(nx+2*npadx)*(ny+2*npady);
  Nc=(nx+2*npadx)*((ny+2*npady)/2+1);
  fftw_init_threads();
  fftw_plan_with_nthreads(NTH);
  fhwp->plan6b=fftw_plan_many_dft_c2r(2,aux,6,(fftw_complex *)(dat),NULL,1,Nc,(double *) (dat), NULL, 1, 2*Nc, FFTW_ESTIMATE);
  fhwp->plan2f=fftw_plan_many_dft_r2c(2,aux,2,(double *)(&dat[6*Nc]),NULL,1,2*Nc,(fftw_complex *) (&dat[6*Nc]), NULL, 1, Nc, FFTW_ESTIMATE);
}

void fhwak(complex *y, complex *dydt){
  int nx=fhwp->nx,ny=fhwp->ny;
  int N=nx*ny;
  int Nc=nx*(ny/2+1);
  int Npad=(nx+2*fhwp->npadx)*(ny+2*fhwp->npady);
  int Npadc=(nx+2*fhwp->npadx)*((ny+2*fhwp->npady)/2+1);
  double *kx=fhwp->kx, *ky=fhwp->ky;
  complex *linmat=fhwp->linmat;
  complex *phik=&y[0];
  complex *nk=&y[Nc];
  complex *dphikdt=&dydt[0];
  complex *dnkdt=&dydt[Nc];
  
  complex *dxphik=&fhwp->dat[0];
  complex *dyphik=&fhwp->dat[Npadc];
  complex *dxnk=&fhwp->dat[2*Npadc];
  complex *dynk=&fhwp->dat[3*Npadc];
  complex *dxwk=&fhwp->dat[4*Npadc];
  complex *dywk=&fhwp->dat[5*Npadc];
  complex *nlphik=&fhwp->dat[6*Npadc];
  complex *nlnk=&fhwp->dat[7*Npadc];
  
  double *dxphi=(double *)&fhwp->dat[0];
  double *dyphi=(double *)&fhwp->dat[Npadc];
  double *dxn=(double *)&fhwp->dat[2*Npadc];
  double *dyn=(double *)&fhwp->dat[3*Npadc];
  double *dxw=(double *)&fhwp->dat[4*Npadc];
  double *dyw=(double *)&fhwp->dat[5*Npadc];
  double *nlphi=(double *)&fhwp->dat[6*Npadc];
  double *nln=(double *)&fhwp->dat[7*Npadc];
  complex i=csqrt(-1.0);
  double ksqr;
  int lx,ly,l,lpad,lsx,lsy;

  memset(&fhwp->dat[0],0,sizeof(complex)*Npadc*8);

#pragma omp parallel shared(nx,ny,dxphik,dyphik,dxwk,dywk,dxnk,dynk,dphikdt,dnkdt,linmat,phik,nk,kx,ky,fhwp,i) private(lx,ly,l,lsx,lsy,lpad,ksqr) num_threads(NTH)
  {
#pragma omp for
        for (l=0;l<Nc;l++){
//    for(lx=0;lx<nx;lx++){
//      for(ly=0;ly<ny/2+1;ly++){
        lx=l/(ny/2+1);
        ly=l%(ny/2+1);
//        l=ly+(ny/2+1)*lx;
	lsy=ly;
	lsx=lx+((int)(2*lx/nx))*(2*fhwp->npadx);
	lpad=lsy+((ny+2*fhwp->npady)/2+1)*lsx;
	ksqr=kx[l]*kx[l]+ky[l]*ky[l];
	dphikdt[l]=linmat[l*4]*phik[l]+linmat[l*4+1]*nk[l]+fhwp->fk_phi[l];
	dnkdt[l]=linmat[l*4+2]*phik[l]+linmat[l*4+3]*nk[l]+fhwp->fk_n[l];
	dxphik[lpad]=i*kx[l]*phik[l];
	dyphik[lpad]=i*ky[l]*phik[l];
	dxnk[lpad]=i*kx[l]*nk[l];
	dynk[lpad]=i*ky[l]*nk[l];
	dxwk[lpad]=-i*kx[l]*ksqr*phik[l];
	dywk[lpad]=-i*ky[l]*ksqr*phik[l];
//      }
    }
  }
  fftw_execute(fhwp->plan6b);
  
#pragma omp parallel shared(nx,ny,dxphi,dyphi,dxw,dyw,dxn,dyn,nlphi,nln,N,Npad,fhwp) private(lx,ly,l) num_threads(NTH)
  {
#pragma omp for
    for(lx=0;lx<(nx+2*fhwp->npadx);lx++){
      for(ly=0;ly<(ny+2*fhwp->npady);ly++){
	l=ly+((ny+2*fhwp->npady)/2+1)*2*lx;
	nlphi[l]=(dxphi[l]*dyw[l]-dyphi[l]*dxw[l])/N/Npad; // or Npad?
	nln[l]=(dxphi[l]*dyn[l]-dyphi[l]*dxn[l])/N/Npad;
      }
    }
  }
  fftw_execute(fhwp->plan2f);
#pragma omp parallel shared(nx,ny,nlphik,nlnk,dphikdt,dnkdt,kx,ky,fhwp) private(lx,ly,l,lsx,lsy,lpad,ksqr) num_threads(NTH)
  {
#pragma omp for
        for (l=0;l<Nc;l++){
//    for(lx=0;lx<nx;lx++){
//      for(ly=0;ly<ny/2+1;ly++){
        lx=l/(ny/2+1);
        ly=l%(ny/2+1);
//        l=ly+(ny/2+1)*lx;
	lsy=ly;
	lsx=lx+((int)(2*lx/nx))*(2*fhwp->npadx);
	lpad=lsy+((ny+2*fhwp->npady)/2+1)*lsx;
	ksqr=kx[l]*kx[l]+ky[l]*ky[l];
	//	l=ly+(ny/2+1)*lx;
	dphikdt[l]+=nlphik[lpad]/ksqr;
	dnkdt[l]-=nlnk[lpad];
      }
//    }
  }
  dphikdt[0]=0.0;
  dnkdt[0]=0.0;
}

void fhw_pad(complex *in,complex *out){
  int nx=fhwp->nx,ny=fhwp->ny;
  int Npadc=(nx+2*fhwp->npadx)*((ny+2*fhwp->npady)/2+1);
  int lx,ly,l,lpad,lsx,lsy;
  memset(out,0,sizeof(complex)*Npadc);
  for(lx=0;lx<nx;lx++){
    for(ly=0;ly<ny/2+1;ly++){
      l=ly+(ny/2+1)*lx;
      lsy=ly;
      lsx=lx+((int)(2*lx/nx))*(2*fhwp->npadx);
      lpad=lsy+((ny+2*fhwp->npady)/2+1)*lsx;
      out[lpad]=in[l];
    }
  }
}
