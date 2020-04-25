#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:16:48 2020

@author: Ozgur D. Gurcan
"""

import numpy as np
import numpy.ctypeslib as ctl
import pyfftw as pyfw
import h5py as h5
import scipy.integrate as spi
import time
import os

nx,ny=1024,1024
Lx,Ly=16*np.pi,16*np.pi
pars={'C':0.1,'kap':0.2,'nu':1e-3,'D':1e-3,'nuZF':0.0,'DZF':0.0,'modified':False,'random_forcing':False}
tvars={'t0':0.0,'t1':1000.0,'dt':1e-2,'dtr':1e-1,'dtout':1.0}
#print(__file__)

pth=os.path.dirname(__file__)
fhw=ctl.load_library('libfhwak.so',pth+'/fhwak')

def set_res(inx,iny):
    global nx,ny,Nx,Ny,npadx,npady
    nx,ny=inx,iny
    Nx,Ny=int(2*np.floor(nx/3)),int(2*np.floor(ny/3))
    npadx,npady=int(np.int_(np.ceil(Nx/4))),int(np.int_(np.ceil(Ny/4)))
    npadx=(int)(np.ceil((2**np.ceil(np.log2(Nx+2*npadx))-Nx)/2))
    npady=(int)(np.ceil((2**np.ceil(np.log2(Ny+2*npady))-Ny)/2))

def f_phik0(kx,ky):
    A=1.0;
    kx0=0.0;
    ky0=0.0;
    sigkx=0.5;
    sigky=0.5;
    th=np.random.rand(kx.shape[0],kx.shape[1])*2*np.pi;
    phik0=A*np.exp(-(kx-kx0)**2/2/sigkx**2-(ky-ky0)**2/2/sigky**2)*np.exp(1j*th);
    return phik0

def fhwak_python(t,y):
    global kx,ky,ksqr,linmat,dat,fftw_obj_dat6b,fftw_obj_dat2f,N,Npad
    phik,nk=y.view(dtype=complex).reshape(2,Nx,int(Ny/2+1))
    dphikdt=linmat[:,:,0,0]*phik+linmat[:,:,0,1]*nk
    dnkdt=linmat[:,:,1,0]*nk+linmat[:,:,1,1]*phik
    dat[:6,npadx:-npadx,:-npady]=np.stack((1j*kx*phik,1j*ky*phik,1j*kx*nk,1j*ky*nk,-1j*kx*ksqr*phik,-1j*ky*ksqr*phik))
    fftw_obj_dat6b()
    dxphi,dyphi,dxn,dyn,dxw,dyw=dat[:6,:,:].view(dtype=float)[:,:,:-2]
    dat[6:,:,:].view(dtype=float)[:,:,:-2]=np.stack(((dxphi*dyw-dyphi*dxw)/N/Npad,(dxphi*dyn-dyphi*dxn)/N/Npad))
    fftw_obj_dat2f()
    dphikdt[ksqr>0]+=dat[6,npadx:-npadx,:-npady][ksqr>0]/ksqr[ksqr>0];
    dnkdt+=-dat[7,npadx:-npadx,:-npady];
    dydt=np.stack((dphikdt,dnkdt)).ravel().view(dtype=float)
    return dydt

def fhwak_c(t,y):
    global dydt
    fhw.fhwak(y.ctypes,dydt.ctypes)
    return dydt


def run(flname,wecontinue=False,fc_phi=f_phik0,fc_n=f_phik0, nuf = lambda nu,ksqr : nu*ksqr, atol=1e-12,rtol=1e-6,ncpu=8, pure_python=False, save_preview=True, prev_size=10):
    global kx,ky,ksqr,linmat,dat,fftw_obj_dat6b,fftw_obj_dat2f,N,Npad,r,dydt
    set_res(nx,ny)
    dat=pyfw.empty_aligned((8,int(Nx+2*npadx),int((Ny+2*npady)/2+1)),dtype=complex);
    dat.fill(0)
    indsx=np.int_(np.round(np.fft.fftfreq(Nx)*Nx))
    indsy=np.arange(0,int(Ny/2+1))
    dkx,dky=2*np.pi/Lx,2*np.pi/Ly
    kx,ky=np.meshgrid(indsx*dkx,indsy*dky,indexing='ij')
    ksqr=(kx**2+ky**2)
    linmat=np.zeros((Nx,int(Ny/2+1),2,2),dtype=complex)
    N=Nx*Ny
    Npad=nx*ny
    if(wecontinue==True):
        fl=h5.File(flname,"r+")
        for l in pars:
            pars[l]=fl['pars/'+l]
        for l in tvars:
            tvars[l]=fl['pars/'+l]
        phires=fl["fields/phi"]
        nres=fl["fields/n"]
        tres=fl["fields/t"]
        tvars['t0']=tres[-1]
        phi0=phires[-1,:,:]
        nf0=nres[-1,:,:]
        phik0=pyfw.empty_aligned((Nx,int(Ny/2+1)),'complex');
        nk0=pyfw.empty_aligned((Nx,int(Ny/2+1)),'complex');
        fftw_objf = pyfw.FFTW(phi0.copy(), phik0, axes=(0, 1))
        phik0=fftw_objf(phi0.copy(),phik0)
        nk0=fftw_objf(nf0.copy(),nk0)
        u0=np.stack((phik0,nk0),0)
        fftw_obj = pyfw.FFTW(phik0.copy(), phi0, axes=(0, 1),direction='FFTW_BACKWARD')
        i=phires.shape[0]
        grp=fl["fields"]
    else:
        phik0=fc_phi(kx,ky)
        phi0=pyfw.empty_aligned(phik0.view(dtype=float)[:,:-2].shape,'float');
        fftw_obj = pyfw.FFTW(phik0.copy(), phi0, axes=(0, 1),direction='FFTW_BACKWARD')
        phi0=fftw_obj();
        nk0=fc_n(kx,ky)
        nf0=pyfw.empty_aligned(nk0.view(dtype=float)[:,:-2].shape,'float');
        nf0=fftw_obj(nk0.copy(),nf0);
        u0=np.stack((phik0,nk0),0)
        fftw_obj = pyfw.FFTW(phik0.copy(), phi0, axes=(0, 1),direction='FFTW_BACKWARD')
        i=0;
        if os.path.exists(flname):
            os.remove(flname)
        fl=h5.File(flname,"w")
        gpars=fl.create_group("pars")
        for l in pars:
            gpars.create_dataset(l,data=pars[l])
        for l in tvars:
            gpars.create_dataset(l,data=tvars[l])
        grp=fl.create_group("fields")
        grp.create_dataset("kx",data=kx)
        grp.create_dataset("ky",data=ky)
        phires=grp.create_dataset("phi",(1,Nx,Ny),maxshape=(None,Nx,Ny),dtype=float)
        nres=grp.create_dataset("n",(1,Nx,Ny),maxshape=(None,Nx,Ny),dtype=float)
        tres=grp.create_dataset("t",(1,),maxshape=(None,),dtype=float)
    if(save_preview):
        prn,prex=os.path.splitext(flname)
        prfln=prn+'_prev'+prex
        if os.path.exists(prfln):
            os.remove(prfln)
        pfl=h5.File(prfln,"w")
        pgrp=pfl.create_group("fields")
        pgrp.create_dataset("kx",data=kx)
        pgrp.create_dataset("ky",data=ky)
        phipr=pgrp.create_dataset("phi",(prev_size,Nx,Ny),dtype=float)
        npr=pgrp.create_dataset("n",(prev_size,Nx,Ny),dtype=float)
        tpr=pgrp.create_dataset("t",(prev_size,),dtype=float)
    C=pars['C']
    kap=pars['kap']
    nu=pars['nu']
    D=pars['D']
    nuZF=pars['nuZF']
    DZF=pars['DZF']
    linmat[ksqr>0,0,0]=-C/(ksqr[ksqr>0])-nuf(nu,ksqr)[ksqr>0]
    linmat[ksqr>0,0,1]=C/(ksqr[ksqr>0])
    linmat[:,:,1,0]=-1j*kap*ky+C
    linmat[:,:,1,1]=-C-nuf(D,ksqr)
    if(pars['modified']):
        linmat[:,0,0,0]=-nuZF
        linmat[:,0,0,1]=0.0
        linmat[:,0,1,0]=0.0
        linmat[:,0,1,1]=-DZF

    if(pure_python):
        fftw_obj_dat6b = pyfw.FFTW(dat[:6,:,:], dat[:6,:,:].view(dtype=float)[:,:,:-2], axes=(1, 2),direction='FFTW_BACKWARD',threads=ncpu)
        fftw_obj_dat2f = pyfw.FFTW(dat[6:,:,:].view(dtype=float)[:,:,:-2],dat[6:,:,:], axes=(1, 2),direction='FFTW_FORWARD',threads=ncpu)
        fhwak=fhwak_python
    else:
        dydt=np.zeros(Nx*(int(Ny/2+1))*4)
        fk_phi=np.zeros_like(phik0)
        fk_n=np.zeros_like(nk0)
        fhw.fhwak_init(Nx,Ny,npadx,npady,dat.ctypes,linmat.ctypes,kx.ctypes,ky.ctypes,fk_phi.ctypes,fk_n.ctypes)
        fhwak=fhwak_c
    dt=tvars['dt']
    dtout=tvars['dtout']
    dtr=tvars['dtr']
    t0=tvars['t0']
    t1=tvars['t1']
    r=spi.RK45(fhwak,t0,u0.view(dtype=float).ravel(),t1,max_step=dt,atol=atol,rtol=rtol)
    epst=1e-12
    ct=time.time()
    toldr=-1.0e12
    toldout=-1.0e12
    while(r.status=='running'):
        told=r.t
        if(r.t>=toldout+dtout-epst and r.status=='running'):
            toldout=r.t
            print("t=",r.t);
            phik,nk=r.y.view(dtype=complex).reshape(2,Nx,int(Ny/2+1))
            phires.resize((i+1,Nx,Ny))
            nres.resize((i+1,Nx,Ny))
            tres.resize((i+1,))
            phik=phik.squeeze()
            nk=nk.squeeze()
            phi=pyfw.empty_aligned((Nx,Ny),'float');
            nf=pyfw.empty_aligned((Nx,Ny),'float');
            phi=fftw_obj(phik.copy(),phi);
            nf=fftw_obj(nk.copy(),nf);
            phires[i,:,:]=phi
            nres[i,:,:]=nf
            tres[i]=r.t
            fl.flush()
            if(save_preview and i>prev_size):
                phipr[:,:,:]=phires[i-prev_size:i,:,:]
                npr[:,:,:]=nres[i-prev_size:i,:,:]
                tpr[:]=tres[i-prev_size:i]
                pfl.flush()
            i=i+1
            print(time.time()-ct,"seconds elapsed.")
        if(r.t>=toldr+dtr-epst and r.status=='running' and pars['random_forcing']==True):
            toldr=r.t
#            force_update()
#            linmat_update()
        while(r.t<told+dt-epst and r.status=='running'):
            res=r.step()
    fl.close()
