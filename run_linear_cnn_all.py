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


import dsutils
import deconv


def get_fname(modelname,npsf,sigma,img):
    return '{}_PSF{}_sigma{:1.3f}_{}'.format(modelname,npsf,sigma,img)

def get_callback_string(fname):
    return 'models/'+fname+'e{epoch:02d}-loss{val_loss:.4f}.hd5'
    

# set seed
np.random.seed(1985) 

#%% create network model
szi=32
lst_size=[[1,17]]
          
#model=dsutils.get_dnn_model(szi,lst_size)
model=dsutils.Sequential()
model.add(dsutils.Convolution2D(lst_size[0][0], lst_size[0][1], lst_size[0][1], border_mode='valid',activation='linear',init='zero', input_shape=(1,szi,szi)))
modelname=dsutils.get_dnn_name(szi,lst_size)
szo=dsutils.get_dnn_szo(szi,lst_size)

sgd = dsutils.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse')

#%%  data parameters

sigma=0.01
npsf=64
nr=5
napp=100000
nval=50000


PSF=deconv.get_PSF_airy(npsf,nr)

#%% load dataset

lst_img=['M31','Hoag','M51a','M81','M101','M104']
#lst_img=['M101']
nb_img=len(lst_img)

I0_all=[deconv.load_fits_image(img) for img in lst_img]

# pre-process images (pste positive and max to 1)
I0_all=[np.maximum(0,I/I.max()) for I in I0_all]
# compute convolution
I0_all_conv=[deconv.fftconvolve(I,PSF,'same') for I in I0_all]
I_all=[I+sigma*np.random.randn(I.shape[0],I.shape[1]) for I in I0_all_conv]

mse_conv=np.array([deconv.mse(I0_all[i],I0_all_conv[i]) for i in range(nb_img)])
mse_convnoise=np.array([deconv.mse(I0_all[i],I_all[i]) for i in range(nb_img)])

#%%

#pl.figure(1)
#for i in range(len(lst_img)):
#    pl.subplot(2,3,i+1)
#    pl.imshow(I0_all[i],interpolation='nearest')
#
#
#pl.figure(2)
#for i in range(len(lst_img)):
#    pl.subplot(2,3,i+1)
#    pl.imshow(I0_all[i],interpolation='nearest')
#
#
#
#pl.figure(3)
#for i in range(len(lst_img)):
#    pl.subplot(2,3,i+1)
#    pl.imshow(I_all[i],interpolation='nearest')

#%% generating dataset

for itest in [3,]:#range(nb_img):
#for itest in [4,]:    
    
    
    print("Image: {:8s}\n===============\n".format(lst_img[itest]))
    # get dataset
    I0test=I0_all[itest]
    Itest=I_all[itest]
    I0app=[I for i,I in enumerate(I0_all) if not i==itest]
    Iapp=[I for i,I in enumerate(I_all) if not i==itest]
    
    fname=get_fname(modelname,npsf,sigma,lst_img[itest])
    path_cb=get_callback_string(fname)
    
    xapp,yapp=dsutils.get_data_multi(Iapp,I0app,szi,szo,napp)
    xval,yval=dsutils.get_data_multi(Iapp,I0app,szi,szo,nval)
    xtest,ytest=dsutils.get_data(Itest,I0test,szi,szo,nval)
    
    # initialize model

    model=dsutils.get_dnn_model(szi,lst_size)
    model.compile(optimizer=sgd, loss='mse')
    
    
    
    
    # callbacks
    earlystop=dsutils.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    saveweights=dsutils.ModelCheckpoint(path_cb, monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    
    
    # print fnfos
    mseapp_conv=(np.sum(mse_conv)-mse_conv[itest])/(nb_img-1)  
    mseapp_convnoise=(np.sum(mse_convnoise)-mse_convnoise[itest])/(nb_img-1) 
    
    print('MSE_conv:      {:e}\nMSE_convnoise: {:e}'.format(mseapp_conv,mseapp_convnoise))
    
    # learning
    nb_epoch=30
    batch_size=50
    history = model.fit(xapp,yapp,
                        shuffle=True,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(xval, yval),
                        callbacks=[earlystop,saveweights])
    
    dsutils.save_model(model,'models/'+fname)
    
    mse=np.mean((ytest-model.predict(xtest))**2)
    print('Test performances:\nMSE_conv:      {:e}\nMSE_convnoise: {:e}\nMSE_test     : {:e}'.format(mse_conv[itest],mse_convnoise[itest],mse))