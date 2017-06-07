# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:50:47 2016

@author: rflamary
"""
import numpy as np
import scipy as sp

import skimage as ski
import skimage.restoration

#from pymoresane.iuwt_convolution import fft_convolve
from scipy.signal import fftconvolve
from scipy.signal import convolve2d as conv2

from astropy.io.fits import getdata


#%% tic/toc
import time
_tictocstart=time.time()


def tic():
    global _tictocstart
    _tictocstart=time.time()

def toc(str_print=''):
    global _tictocstart
    tp=time.time()-_tictocstart
    print('{}{:1.3f}s elapsed'.format(str_print,tp))
    return tp

def toq(str_print=''):
    global _tictocstart
    tp=time.time()-_tictocstart
    return tp
#%% utils PSF/image

def get_PSF_airy(n,nr):
    xpsf=np.linspace(-1,1,n)
    Xg,Yg=np.meshgrid(xpsf,xpsf)
    R=np.sqrt(Xg**2+Yg**2)*np.pi*nr
    PSF=(sp.special.j1(R)/R)**2
    PSF=PSF/PSF.sum()
    return PSF




def load_fits_image(name):
    path='data/{}.fits'.format(name)

    dat = getdata(path)
    #pl.imshow(dat)
    return 1.0*dat


def crop(I,cr):
    return I[cr:-cr,cr:-cr]

def mse_crop(I1,I2,cr):
    return np.mean((crop(I1,cr)-crop(I2,cr))**2)

def mse(I1,I2):
    return np.mean((I1-I2)**2)
#%% decon algorithms

def richardson_lucy(image, psf, itermax=50):

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for _ in range(itermax):
        relative_blur = image / sp.signal.fftconvolve(im_deconv, psf, 'same')
        im_deconv *= sp.signal.fftconvolve(relative_blur, psf_mirror, 'same')


    return im_deconv

def richardson_lucy_path(image, psf,imagetrue, itermax=50,cr=1):

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]
    im_deconv_best=im_deconv;
    err=np.zeros((itermax,))


    for i in range(itermax):
        relative_blur = image / sp.signal.fftconvolve(im_deconv, psf, 'same')
        im_deconv *= sp.signal.fftconvolve(relative_blur, psf_mirror, 'same')
        err[i]=mse_crop(im_deconv,imagetrue,cr)
        if err[i]==err.min():
            im_deconv_best=im_deconv.copy()

    return im_deconv_best,err

def total_variation(x):

    uh = np.zeros_like(x)
    uh[:,:-1] = x[:,1:] - x[:,:-1]

    uv = np.zeros_like(x)
    uv[:-1,:] = x[1:,:] - x[:-1,:]

    return np.sum(np.sqrt(uh**2 + uv**2))


def vc_recons_tv(imgf, psf, mu=1e-4, x = None, itermax = 1000,compcost=False, sky=None):
# A Generic Proximal Algorithm for Convex Optimization—Application to Total Variation Minimization.
# Condat, L. (2014). IEEE Signal Processing Letters, 21(8), 985–989.

    sigmac = mu
    tauc = 0.99/(0.5*10  + 8.0*sigmac)    # ok for gaussian psf normalized to 1

    if x==None:
        x=imgf

    psfadj = psf[::-1,::-1]

    nx, ny = imgf.shape
    u1 = np.zeros_like(x)
    u2 = np.zeros_like(x)
    conv=lambda x,psf: sp.signal.fftconvolve(x, psf, 'same')
    if compcost:
        cost_condat=np.zeros((itermax+1,))
        err_condat=np.zeros((itermax+1,))
        def cost(x):
            return 0.5*np.sum((imgf-conv(x,psf))**2)+mu*total_variation(x)

        cost_condat[0]=cost(x)
        if not sky==None:
            err_condat[0]=np.mean((sky-x)**2)

    for it in range(itermax):

        # compute gradient
        grad=conv(conv(x,psf)-imgf,psfadj)

        # compute adjunct of tv applied to u = (u1, u2)
        lsu1u2=np.hstack((-u1[:,0:1],u1[:,:-1]-u1[:,1:]))+np.vstack((-u2[0:1,:], u2[:-1,:] - u2[1:,:]))

        # update x\tilde
        xt=np.maximum(0, x - tauc*(grad + lsu1u2))

        # update utilde = (utilde 1, utilde 2)
        dx=2*x-xt
        ut1=u1+sigmac*np.hstack((dx[:,1:]-dx[:,:-1],np.zeros((nx,1))))
        ut2=u2+sigmac*np.vstack((dx[1:,:]-dx[:-1,:],np.zeros((1,ny))))

        den=np.maximum(1.0,np.sqrt(ut1**2+ut2**2))
        ut1=ut1/den
        ut2=ut2/den

        u1=ut1
        u2=ut2

        x=xt

        if compcost:
            cost_condat[it+1]=cost(x)
            if not sky==None:
                err_condat[it+1]=np.mean((sky-x)**2)

    if compcost:
        return x,cost_condat,err_condat
    else:
        return x



def wiener(I,PSF,lambd=1):
    return ski.restoration.wiener(I,PSF,lambd)

def richardson_lucyl(I,PSF,itermax=50):
    return ski.restoration.richardson_lucy(I,PSF,iterations=itermax)

def wiener0(I,PSF,lambd=1):
    fPSF=np.fft.fft2(PSF,I.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(I)*fPSF/(np.abs(fPSF)**2+lambd)))
