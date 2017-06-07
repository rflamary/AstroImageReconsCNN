# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:27:25 2016

@author: rflamary
"""

import sys
import numpy as np
import scipy as sp
import scipy.signal
import scipy.io as spio
import deconv
import matplotlib.pylab as pl
import dsutils

import theano

def get_fname(method,n,npsf,sigma,img):
    return 'res/{}_{}x{}_PSF{}_sigma{:1.3f}_{}.mat'.format(method,n,n,npsf,sigma,img)

def get_fname_all(method,n,npsf,sigma):
    return 'res/{}_{}x{}_PSF{}_sigma{:1.3f}_all.mat'.format(method,n,n,npsf,sigma)


#%% load image

I0=deconv.load_fits_image('M51a')
I0=I0/I0.max()
    
#%% generat
i=2
cr=32
lst_img=['M31','Hoag','M51a','M81','M101','M104']
#lst_img=['M31']
nb_img=len(lst_img)

def sel(I):
    return I[300:-cr-100,300:-cr-100]

img_txt=lst_img[i]
method='none'

I0=deconv.load_fits_image(img_txt) 
n=1024
iv=1200;jv=1200
I0=I0[iv:iv+n,jv:jv+n]

npsf=64
sigma=0.01


fname=get_fname('none',n,npsf,sigma,img_txt)
data=spio.loadmat(fname)
Inoise=data['Irec']


#%% get PSF
npsf=64
nr=5


#%% deconvnn

fname='models/32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_M51a'

model=dsutils.load_model(fname)


model.compile(optimizer='SGD', loss='mse')


sz=32
szp=14
deconv.tic()
I_dcnnn=dsutils.apply_model(Inoise,model,sz,szp)
deconv.toc()


#%% visu last layer

szp2=18
Ip=dsutils.im2patch(Inoise,sz,szp2)

convout1_f = theano.function([model.get_input_at(0)], model.layers[1].get_output_at(0),allow_input_downcast=True)

Ip2=convout1_f(Ip)

I_layer1=dsutils.patch2im(Ip2[:,0:1,:,:],Inoise.shape,sz,szp2)

#
#pl.figure("last layer")
#
#for i in range(16):
#    pl.subplot(4,4,i+1)
#    pl.imshow(dsutils.patch2im(Ip2[:,i:i+1,:,:],Inoise.shape,sz,szp2),interpolation='nearest')
    
Il2_1=dsutils.patch2im(Ip2[:,15:16,:,:],Inoise.shape,sz,szp2)
Il2_2=dsutils.patch2im(Ip2[:,2:3,:,:],Inoise.shape,sz,szp2)    
Il2_3=dsutils.patch2im(Ip2[:,7:8,:,:],Inoise.shape,sz,szp2)    

#%% visu first layer
szp2=23
Ip=dsutils.im2patch(Inoise,sz,szp2)

convout0_f = theano.function([model.get_input_at(0)], model.layers[0].get_output_at(0),allow_input_downcast=True)

Ip2=convout0_f(Ip)

I_layer1=dsutils.patch2im(Ip2[:,0:1,:,:],Inoise.shape,sz,szp2)


#pl.figure("first layer")
#
#for i in range(64):
#    pl.subplot(8,8,i+1)
#    pl.imshow(dsutils.patch2im(Ip2[:,i:i+1,:,:],Inoise.shape,sz,szp2),interpolation='nearest')

Il1_1=dsutils.patch2im(Ip2[:,4:5,:,:],Inoise.shape,sz,szp2)
Il1_2=dsutils.patch2im(Ip2[:,2:3,:,:],Inoise.shape,sz,szp2)
Il1_3=dsutils.patch2im(Ip2[:,33:34,:,:],Inoise.shape,sz,szp2)

#%%

yt=1
fs=10

pl.figure(1)

pl.subplot(3,3,1)
pl.imshow(sel(Il1_1),cmap='gray')
pl.title('Layer 1 output 1',fontsize=fs,y=yt)
pl.axis("off")

pl.subplot(3,3,2)
pl.imshow(sel(Il1_2),cmap='gray')
pl.title('Layer 1 output 2',fontsize=fs,y=yt)
pl.axis("off")

pl.subplot(3,3,3)
pl.imshow(sel(Il1_3),cmap='gray')
pl.title('Layer 1 output 3',fontsize=fs,y=yt)
pl.axis("off")
  
pl.subplot(3,3,4)
pl.imshow(sel(Il2_1),cmap='gray')
pl.title('Layer 2 output 1',fontsize=fs,y=yt)
pl.axis("off")

pl.subplot(3,3,5)
pl.imshow(sel(Il2_2),cmap='gray')
pl.title('Layer 2 output 2',fontsize=fs,y=yt)
pl.axis("off")

pl.subplot(3,3,6)
pl.imshow(sel(Il2_3),cmap='gray')
pl.title('Layer 2 output 3',fontsize=fs,y=yt)
pl.axis("off")

pl.subplots_adjust(wspace=-.5,hspace=0.3)
pl.savefig('imgs/images_features.png',dpi=300,bbox_inches='tight',pad_inches=.01)
    