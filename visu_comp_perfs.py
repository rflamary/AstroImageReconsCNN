"""
Created on Fri Jul 29 14:28:32 2016

@author: rflamary
"""

import sys
import numpy as np
import scipy as sp
import scipy.io as spio
import matplotlib.pylab as pl



def get_fname(method,n,npsf,sigma,img):
    return 'res/{}_{}x{}_PSF{}_sigma{:1.3f}_{}.mat'.format(method,n,n,npsf,sigma,img)

def get_fname_all(method,n,npsf,sigma):
    return 'res/{}_{}x{}_PSF{}_sigma{:1.3f}_all.mat'.format(method,n,n,npsf,sigma)

#%%

sigma=0.01
npsf=64
nr=5
cr=32
n=1024

lst_img=['M31','Hoag','M51a','M81','M101','M104']
#lst_img=['M31']
nb_img=len(lst_img)

lst_methods=['wiener','rl','vc_tv','cnn0','cnn']
nbm=len(lst_methods)

#%% load perfs

MSE=np.zeros((nb_img,nbm))
t_comp=np.zeros((nb_img,nbm))
for i,method in enumerate(lst_methods):
    fname=get_fname_all(method,n,npsf,sigma)
    data=spio.loadmat(fname)
    MSE[:,i]=data['mse'][0][:nb_img]
    t_comp[:,i]=data['t'][0][:nb_img]
    
#%%
    
print('            |'+'|'.join(['{:14s}'.format(method) for method in lst_methods]))
print('') 
for i,img in enumerate(lst_img):
    print('{:12s}|{}'.format(img,'|'.join(['{:1.8e}'.format(MSE[i,j]) for j in range(nbm)])))
print('')   
print('{:12s}|{}'.format('Mean mse','|'.join(['{:1.8e}'.format(np.mean(MSE[:,j])) for j in range(nbm)])))    
print('{:12s}|{}'.format('Mean time','|'.join(['{:1.8e}'.format(np.mean(t_comp[:,j])) for j in range(nbm)]))) 


#%% PSNR
    
print('            |'+'|'.join(['{:10s}'.format(method) for method in lst_methods]))
print('') 
for i,img in enumerate(lst_img):
    print('{:12s}|{}'.format(img,'|'.join(['{:10.2f}'.format(-10*np.log10(MSE[i,j])) for j in range(nbm)])))
print('')   
print('{:12s}|{}'.format('Mean PSNR','|'.join(['{:10.2f}'.format(-10*np.log10(np.mean(MSE[:,j]))) for j in range(nbm)])))    
print('{:12s}|{}'.format('Mean time','|'.join(['{:10.2f}'.format(np.mean(t_comp[:,j])) for j in range(nbm)]))) 

#%% PSNR
    
print('            &'+'&'.join(['{:10s}'.format(method) for method in lst_methods]))
print('') 
for i,img in enumerate(lst_img):
    print('{:12s}&{}'.format(img,'&'.join(['{:10.2f}'.format(-10*np.log10(MSE[i,j])) for j in range(nbm)])))
print('')   
print('{:12s}&{}'.format('Mean PSNR','&'.join(['{:10.2f}'.format(-10*np.log10(np.mean(MSE[:,j]))) for j in range(nbm)])))    
print('{:12s}&{}'.format('Mean time','&'.join(['{:10.2f}'.format(np.mean(t_comp[:,j])) for j in range(nbm)]))) 


