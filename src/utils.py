#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:40:11 2018

@author: ogurcan
"""
import numpy as np
import h5py as h5
import pyfftw as pyfw
import os
import subprocess as sbp

tmpdir='pyhw_tempdir'
eps=1e-20

def get_spec(i,ntav):
    global fftw_objf,phik0,nk0,kx,ky,k,N,kn,dkx,dky,Nx,Ny,phires,nres
    phi0=phires[i:i+ntav,:,:]
    nf0=nres[i:i+ntav,:,:]
    phik0=fftw_objf(phi0,phik0)
    nk0=fftw_objf(nf0,nk0)
    Ek0=np.fft.fftshift(np.mean(np.abs(phik0)**2,0)*(kx**2+ky**2),0)
    Fk0=np.fft.fftshift(np.mean(np.abs(nk0)**2,0),0)
    En=np.zeros(N)
    Fn=np.zeros(N)
    for l in range(N-1):
        En[l]=np.sum(Ek0[(k>=kn[l]) & (k<kn[l+1])])*dkx*dky/Nx**2/Ny**2
        Fn[l]=np.sum(Fk0[(k>=kn[l]) & (k<kn[l+1])])*dkx*dky/Nx**2/Ny**2    
    return En,Fn

def spec(flname,ntav=2,nt0=-1):
    global fl,fftw_objf,phik0,nk0,kx,ky,k,N,kn,dkx,dky,Nx,Ny,Nt,phires,nres
    if (':' in flname):
        flname=sync_rf(flname)
    fl=h5.File(flname,"r")
    phires=fl["fields/phi"]
    nres=fl["fields/n"]
    kx,ky=fl["fields/kx"][:],fl["fields/ky"][:]
    
    dkx=kx[1,0]-kx[0,0]
    dky=ky[0,1]-ky[0,0]
    
    Nt=phires.shape[0]
    Nx=phires.shape[1]
    Ny=phires.shape[2]

    k=np.sqrt(np.fft.fftshift(kx,0)**2+np.fft.fftshift(ky,0)**2)
    kmin=0.0
    kmax=np.max(kx)+1.0
    N=300
    dk=(kmax-kmin)/N
    kn=np.arange(kmin,kmax,dk)

    if (nt0<0):
        nt0=phires.shape[0]-ntav+nt0+1    

    phi0=phires[nt0:nt0+ntav,:,:]
#    nf0=nres[nt0:nt0+ntav,:,:]
    phik0=pyfw.empty_aligned((ntav,Nx,int(Ny/2+1)),'complex');
    nk0=pyfw.empty_aligned((ntav,Nx,int(Ny/2+1)),'complex');
    fftw_objf = pyfw.FFTW(phi0, phik0, axes=(1, 2))

    En,Fn=get_spec(nt0,ntav)
    
    return En,Fn,kn

def sync_rf(flname):
    if(not os.path.exists(tmpdir)) : os.mkdir(tmpdir)
    flname_orig=flname
    sbp.call(['rsync','-havuP',flname,'./'+tmpdir+'/'])
    return tmpdir+'/'+os.path.basename(flname_orig)

def do_plot(En,Fn,kn,fkn1,fkn2,lab1,lab2,ax):
    qd=ax.loglog(kn[En>eps],En[En>eps],'x-',kn[Fn>eps],Fn[Fn>eps],'+-')
    kr=np.arange(3,50)
    ax.loglog(kn[kr],fkn1(kn[kr]),'k--')
    ax.loglog(kn[kr],fkn2(kn[kr]),'k--')
    ax.legend([lab1,lab2],fontsize=14)
    ax.text(kn[kr[-10]],fkn1(kn[kr[-10]])/2,'$k^{-3}$',fontsize=14)
    ax.text(kn[kr[-10]],fkn2(kn[kr[-10]])/2,'$k^{-1}$',fontsize=14)
    return qd
    
def plot_spec(flname,ntav=2,nt0=-1,fkn1 = lambda k : 1e-4*k**(-3), fkn2 = lambda k : 1e-3*k**(-1), lab1='$E(k)$', lab2='$F(k)$'):
    global fl
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pylab as plt
    if (':' in flname):
        flname=sync_rf(flname)
    En,Fn,kn=spec(flname,ntav,nt0)
    do_plot(En,Fn,kn,fkn1,fkn2,lab1,lab2,ax=plt.gca())
    kr=np.arange(3,50)
    plt.loglog(kn[En>eps],En[En>eps],'x-',kn[Fn>eps],Fn[Fn>eps],'+-')
    plt.loglog(kn[kr],fkn1(kn[kr]),'k--')
    plt.loglog(kn[kr],fkn2(kn[kr]),'k--')
    plt.legend([lab1,lab2],fontsize=14)
    plt.text(kn[kr[-10]],fkn1(kn[kr[-10]])/2,'$k^{-3}$',fontsize=14)
    plt.text(kn[kr[-10]],fkn2(kn[kr[-10]])/2,'$k^{-1}$',fontsize=14)
    fl.close()

def update_spec_anim(j,qd,phi,n,ntav):
    print(j)
    En,Fn=get_spec(j,ntav)
    qd[0].set_xdata(kn[En>eps])
    qd[0].set_ydata(En[En>eps])
    qd[1].set_xdata(kn[Fn>eps])
    qd[1].set_ydata(Fn[Fn>eps])
    return qd

def spec_anim(flname,outfl,vmin=1e-8,vmax=1e-2,ntav=2):
    global fl,Nt
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pylab as plt
    import matplotlib.animation as anim
    if (':' in flname):
        flname=sync_rf(flname)
    En,Fn,kn=spec(flname,ntav)
    w, h = plt.figaspect(0.8)
    fig,ax=plt.subplots(1,1,sharey=True,figsize=(w,h))
    qd=ax.loglog(kn[En>eps],En[En>eps],'x-',kn[Fn>eps],Fn[Fn>eps],'+-')
    kr=np.arange(3,50)
    ax.loglog(kn[kr],1e-4*kn[kr]**(-3),'k--')
    ax.loglog(kn[kr],1e-3*kn[kr]**(-1),'k--')
    ax.legend(['$E(k)$','$F(k)$'],fontsize=14)
    ax.text(kn[kr[-10]],3e-4*kn[kr[-10]]**(-3),'$k^{-3}$',fontsize=14)
    ax.text(kn[kr[-10]],3e-3*kn[kr[-10]]**(-1),'$k^{-1}$',fontsize=14)
    ax.axis([kn[1]-eps,kn[-1],vmin,vmax])
    ani = anim.FuncAnimation(fig, update_spec_anim, interval=0, frames=Nt-ntav, blit=True, fargs=(qd,phires,nres,ntav))
    ani.save(outfl,dpi=200,fps=25)
    fl.close()
    sbp.call(['vlc',outfl])

def update_anim(j,qd,phi,n):
    print(j)
    qd[0].set_array(np.real(phi[j,]).T.ravel())
    qd[1].set_array(np.real(n[j,]).T.ravel())
    return qd

def anim(flname,outfl,vm=1.0,vmn=1.0,ntav=2):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pylab as plt
    import matplotlib.animation as anim
    if (':' in flname):
        flname=sync_rf(flname)
    fl=h5.File(flname,"r")
    phi=fl['fields/phi']
    n=fl['fields/n']
    w, h = plt.figaspect(0.5)
    fig,ax=plt.subplots(1,2,sharey=True,figsize=(w,h))
    qd0 = ax[0].pcolormesh(np.real(phi[0,].T),shading='flat',vmin=-vm,vmax=vm,cmap='seismic',rasterized=True)
    qd1 = ax[1].pcolormesh(np.real(n[0,].T),shading='flat',vmin=-vmn,vmax=vmn,cmap='seismic',rasterized=True)
    fig.tight_layout()
    ax[0].axis('square')
    ax[1].axis('square')
    Nt=phi.shape[0]
    ani = anim.FuncAnimation(fig, update_anim, interval=0, frames=Nt, blit=True, fargs=((qd0,qd1),phi,n))
    ani.save(outfl,dpi=200,fps=25)
    sbp.call(['vlc',outfl])
    fl.close()