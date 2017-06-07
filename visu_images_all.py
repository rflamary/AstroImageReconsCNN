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


#%% load dataset

lst_img=['M31','Hoag','M51a','M81','M101','M104']
#lst_img=['M31']
nb_img=len(lst_img)

I0_all=[deconv.load_fits_image(img) for img in lst_img]

# pre-process images (pste positive and max to 1)
I0_all=[np.maximum(0,I/I.max()) for I in I0_all]
# compute convolution


#%% plot all images

pl.figure(1)
pl.clf()
vmin=0.15
vmax=0.9
k=1

yt=0.99
fs=10
for i in range(6):
    pl.subplot(3,3,1+i)
    if i==5:
        k=0.7
    pl.imshow(I0_all[i]*k,cmap='gray',vmin=vmin,vmax=vmax)
    pl.title(lst_img[i],fontsize=fs,y=yt)
    pl.axis("off")

#pl.subplots_adjust(wspace=-.1,hspace=0.12)
pl.savefig('imgs/images.png',dpi=200,bbox_inches='tight',pad_inches=.005)
