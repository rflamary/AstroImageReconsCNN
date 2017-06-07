# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:02:40 2016

@author: rflamary
"""

import sys
import numpy as np
import scipy as sp
import scipy.io as spio
import matplotlib.pylab as pl

import deconv


def get_fname(method,n,npsf,sigma,img):
    return 'res/{}_{}x{}_PSF{}_sigma{:1.3f}_{}.mat'.format(method,n,n,npsf,sigma,img)

def get_fname_all(method,n,npsf,sigma):
    return 'res/{}_{}x{}_PSF{}_sigma{:1.3f}_all.mat'.format(method,n,n,npsf,sigma)

    

# set seed
np.random.seed(1985) 


if len(sys.argv)>1:
    method=sys.argv[1]
else:
    method='none'



#%% PSF
sigma=0.01
npsf=64
nr=5
cr=32

PSF=deconv.get_PSF_airy(npsf,nr)
npsf=64
#%%

lst_img=['M31','Hoag','M51a','M81','M101','M104']
#lst_img=['M31']
nb_img=len(lst_img)

lst_methods=['wiener','rl','vc_tv','cnn0','cnn']
nbm=len(lst_methods)


#%% generat
i=2

def sel(I):
    return I[300:-cr-100,300:-cr-100]

img_txt=lst_img[i]
method='none'

I0=deconv.load_fits_image(img_txt) 
n=1024
iv=1200;jv=1200
I0=I0[iv:iv+n,jv:jv+n]


fname=get_fname('none',n,npsf,sigma,img_txt)
data=spio.loadmat(fname)
Inoise=data['Irec']

fname=get_fname('wiener',n,npsf,sigma,img_txt)
data=spio.loadmat(fname)
Iwiener=data['Irec']

fname=get_fname('rl',n,npsf,sigma,img_txt)
data=spio.loadmat(fname)
Irl=data['Irec']

fname=get_fname('vc_tv',n,npsf,sigma,img_txt)
data=spio.loadmat(fname)
Itv=data['Irec']
    
fname=get_fname('cnn0',n,npsf,sigma,img_txt)
data=spio.loadmat(fname)
Icnn0=data['Irec']
  
fname=get_fname('cnn',n,npsf,sigma,img_txt)
data=spio.loadmat(fname)
Icnn=data['Irec']

#%% plot

szsb=[4,4]

pl.figure(1)
pl.clf()
fs=5

vmin=0.15
vmax=0.8


pl.subplot(szsb[0],szsb[1],1)
pl.imshow(sel(I0)/sel(I0).max(),cmap='gray',vmin=vmin,vmax=vmax)
pl.title('Original',fontsize=fs,y=0.95)
pl.axis("off")


pl.subplot(szsb[0],szsb[1],2)
pl.imshow(np.sqrt(PSF),cmap='gray',interpolation='nearest')
pl.title('PSF',fontsize=fs,y=0.95)
pl.axis("off")

pl.subplot(szsb[0],szsb[1],3)
pl.imshow(sel(Inoise),cmap='gray',vmin=vmin,vmax=vmax)
pl.title('Conv. + noisy',fontsize=fs,y=0.95)
pl.axis("off")

pl.subplot(szsb[0],szsb[1],4)
pl.imshow(sel(Iwiener),cmap='gray',vmin=vmin,vmax=vmax)
pl.title('Wiener',fontsize=fs,y=0.95)
pl.axis("off")

pl.subplot(szsb[0],szsb[1],5)
pl.imshow(sel(Irl),cmap='gray',vmin=vmin,vmax=vmax)
pl.title('RL',fontsize=fs,y=0.95)
pl.axis("off")

pl.subplot(szsb[0],szsb[1],6)
pl.imshow(sel(Itv),cmap='gray',vmin=vmin,vmax=vmax)
pl.title('Prox. TV',fontsize=fs,y=0.95)
pl.axis("off")

pl.subplot(szsb[0],szsb[1],7)
pl.imshow(sel(Icnn0),cmap='gray',vmin=vmin,vmax=vmax)
pl.title('1-CNN',fontsize=fs,y=0.95)
pl.axis("off")

pl.subplot(szsb[0],szsb[1],8)
pl.imshow(sel(Icnn0),cmap='gray')
pl.title('3-CNN',fontsize=fs,y=0.95)
pl.axis("off")
pl.subplots_adjust(wspace=0.0,hspace=0.15)
pl.savefig('imgs/images_rec.png',dpi=400,bbox_inches='tight',pad_inches=.01)
