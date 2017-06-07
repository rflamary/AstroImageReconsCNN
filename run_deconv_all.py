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
I0_all_conv=[deconv.fftconvolve(I,PSF,'same') for I in I0_all]
I_all=[I+sigma*np.random.randn(I.shape[0],I.shape[1]) for I in I0_all_conv]

mse_conv=np.array([deconv.mse(I0_all[i],I0_all_conv[i]) for i in range(nb_img)])
mse_convnoise=np.array([deconv.mse(I0_all[i],I_all[i]) for i in range(nb_img)])


mse_list=np.zeros((nb_img,))
t_list=np.zeros((nb_img,))

tosave={}
#%% generating dataset

for itest in range(nb_img):
    
    tosave_it={}
    print("\nImage: {:8s}\n===============".format(lst_img[itest]))
    # get dataset
    I0=I0_all[itest]
    I0conv=I0_all_conv[itest]
    I=I_all[itest]
    
    n=1024
    iv=1200;jv=1200
    I0=I0[iv:iv+n,jv:jv+n]
    I=I[iv:iv+n,jv:jv+n]
    
    if method=='none':
        Irec=I
        mse=deconv.mse_crop(Irec,I0,cr)
        t=0
        
    elif method=='wiener':
        
        # parameter validation
        nblambda=100
        lambda_list=np.logspace(-3,5,nblambda)
        mselist_wiener=np.zeros((nblambda,))
        for i,lambd in enumerate(lambda_list):
            mselist_wiener[i]=deconv.mse_crop(I0,deconv.wiener(I,PSF,lambd),cr)
        lambd=lambda_list[mselist_wiener.argmin()]
        tosave_it['mse_list']=mselist_wiener
        tosave_it['lambdalist']=lambda_list
        
        # final estimation
        deconv.tic()
        Irec=deconv.wiener(I,PSF,lambd)
        t=deconv.toq()
        mse=deconv.mse_crop(Irec,I0,cr)
        
    elif method=='rl':    
        itmax=100
        I_rl,mselist_rl=deconv.richardson_lucy_path(I,PSF,I0,itmax)
        tosave_it['mse_list']=mselist_rl
        nit=mselist_rl.argmin()+1
        
        # final estimation
        deconv.tic()
        Irec=deconv.richardson_lucy(I,PSF,nit)
        t=deconv.toq()
        mse=deconv.mse_crop(Irec,I0,cr)        
    elif method=='vc_tv':
        
        nblambda=20
        lambda_list=np.logspace(0,4,nblambda)   
        mselist_vc_tv=np.zeros((nblambda,))   
        for i,lambd in enumerate(lambda_list):
            print('{}/{}'.format(i,20))
            mselist_vc_tv[i]=deconv.mse_crop(I,deconv.vc_recons_tv(I, PSF, lambd, itermax = 500),cr)
        tosave_it['mse_list']=mselist_vc_tv
            
        mu=lambda_list[mselist_vc_tv.argmin()]
        
        # final estimation
        deconv.tic()
        Irec=deconv.vc_recons_tv(I, PSF, mu, x = None, itermax = 500,compcost=False, sky=None)
        t=deconv.toq()
        mse=deconv.mse_crop(Irec,I0,cr)
        
        
    elif method=='cnn':
        import dsutils
        def get_fname2(modelname,npsf,sigma,img):
            return '{}_PSF{}_sigma{:1.3f}_{}'.format(modelname,npsf,sigma,img)
            
        
        szi=32
        lst_size=[[64,10],
          [16,6],
          [1,5]]
          
        modelname=dsutils.get_dnn_name(szi,lst_size)
        szo=dsutils.get_dnn_szo(szi,lst_size)
        
        sigma=0.01
        npsf=64
        nr=5    
        
        
        fname=get_fname2(modelname,npsf,sigma,lst_img[itest])
        model=dsutils.load_model('models/'+fname)
        Irec=dsutils.apply_model(I,model,szi,szo)
        deconv.tic()
        Irec=dsutils.apply_model(I,model,szi,szo)
        t=deconv.toq()
        mse=deconv.mse_crop(Irec,I0,cr)

    elif method=='cnn0':
        import dsutils
        def get_fname2(modelname,npsf,sigma,img):
            return '{}_PSF{}_sigma{:1.3f}_{}'.format(modelname,npsf,sigma,img)
            
        
        szi=32
        lst_size=[[1,17]]
          
        modelname=dsutils.get_dnn_name(szi,lst_size)
        szo=dsutils.get_dnn_szo(szi,lst_size)
        
        sigma=0.01
        npsf=64
        nr=5    
        
        
        fname=get_fname2(modelname,npsf,sigma,lst_img[itest])
        model=dsutils.load_model('models/'+fname)
        Irec=dsutils.apply_model(I,model,szi,szo)
        deconv.tic()
        Irec=dsutils.apply_model(I,model,szi,szo)
        t=deconv.toq()
        mse=deconv.mse_crop(Irec,I0,cr)        
    else:
        mse=0
        t=0
        print("Warning: unknown method")
        break
    
    print('{:20s}\t{:e}'.format('MSE (nodeconv)',deconv.mse_crop(I,I0,cr)))
    print('{:20s}\t{:e}'.format('MSE',mse))
    print('{:20s}\t{:1.3f}s'.format('Time',t))
        
    # save iterations
    fname_it=get_fname(method,n,npsf,sigma,lst_img[itest])
    tosave_it.update({'mse':mse,'t':t,'Irec':Irec})
    spio.savemat(fname_it,tosave_it)
    
    
    mse_list[itest]=mse
    t_list[itest]=t
    # save all
    fname=get_fname_all(method,n,npsf,sigma)
    tosave.update({'mse':mse_list,'t':t_list})
    spio.savemat(fname,tosave)    
    
    
    
    
  
