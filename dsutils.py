# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:17:18 2016

@author: rflamary
"""
from keras.models import model_from_json
import sklearn.feature_extraction
import numpy as np


from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Activation
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,UpSampling2D
from keras.utils import np_utils
from keras.layers import Input, Dense, Lambda
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint,EarlyStopping

#%% keras callbacks
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#%% Kera model saving/loading
def save_model(model,fname='mymodel'):
    model.save_weights(fname+'.h5')
    open(fname+'.json', 'w').write(model.to_json())  
    
def load_model(fname):
    model = model_from_json(open(fname+'.json').read())
    model.load_weights(fname+'.h5')
    return model
    
#%% model creation
    
def get_dnn_szo(szi,lst_size):
    szo=szi
    for i,j in lst_size:
        szo-=j-1
    return szo
    
def get_dnn_model(szi,lst_size,lst_act=None,szf=1):
    """
    Return deconvolutionnal neural network model for input umage (szf,szi,szi)
    
    lstsize is a lits  of length depth of the network
    
    for layer k:
        - lstsize[k][0] : number of output image (nb of filters)
        - lstsize[k][0] : size of the filters  (square filters)
    """
    model=Sequential()
    if lst_act==None:
        lst_act=['relu' for i in lst_size]
    if lst_size:
        model.add(Convolution2D(lst_size[0][0], lst_size[0][1], lst_size[0][1], border_mode='valid',activation=lst_act[0], input_shape=(1,szi,szi)))
    for i in range(1,len(lst_size)):
        model.add(Convolution2D(lst_size[i][0], lst_size[i][1], lst_size[i][1], border_mode='valid',activation=lst_act[i]))
    return model
    
def get_dnn_name(szi,lst_size,lst_act=None):
    if lst_act==None:
        layers='_'.join(['{}x{}-{}'.format(szf,szf,nbf) for nbf,szf in lst_size])
    res='{}x{}_{}'.format(szi,szi,layers)
    return res


#%% Dataset generation
def get_sample(I,Ip,sz,szp):
    """
    geta  random sample (small image from I of size sz*sz and its corresponding 
    smaller image from Ip of size szp*szp)
    """
    i=np.random.randint(0,I.shape[0]-sz)
    j=np.random.randint(0,I.shape[1]-sz)
    delt=int(np.ceil((sz-szp)*1.0/2))
    if szp==1:
        y=Ip[i+delt,j+delt]
    else:
        y=Ip[i+delt:i+sz-delt,j+delt:j+sz-delt]
    return I[i:i+sz,j:j+sz],y


def get_data(I,Ip,sz,szp,n):
    xapp=np.zeros((n,1,sz,sz))
    xtest=np.zeros((n,1,szp,szp))
    for i in range(n):
        xapp[i,:,:,:],xtest[i,:,:,:]=get_sample(I,Ip,sz,szp)
    return xapp,xtest
    
def get_data_multi(I,Ip,sz,szp,n):
    nbi=len(I)
    xapp=np.zeros((n,1,sz,sz))
    xtest=np.zeros((n,1,szp,szp))
    for i in range(n):
        im=np.random.randint(0,nbi)
        xapp[i,:,:,:],xtest[i,:,:,:]=get_sample(I[im],Ip[im],sz,szp)
    return xapp,xtest    
    
#%% image patch utils
def im2patch_sklearn(I,sz):
    temp=sklearn.feature_extraction.image.extract_patches_2d(I,(sz,sz))
    temp=temp.reshape((temp.shape[0],1,temp.shape[1],temp.shape[2]))
    return temp
    
def im2patch(I,sz,szp):
    delt,n1,n2=im2patch_len(I.shape,sz,szp)
    idx=0
    patches=np.zeros((n1*n2,1,sz,sz))
    for i in range(n1):
        for j in range(n2):
            patches[idx,:,:,:]=I[i*szp:i*szp+sz,j*szp:j*szp+sz]
            idx+=1
    return patches    
    
def im2subpatch(I,sz,szp):
    delt,n1,n2=im2patch_len(I.shape,sz,szp)
    patches=np.zeros((n1*n2,1,szp,szp))
    idx=0
    for i in range(n1):
        for j in range(n2):
            patches[idx,:,:,:]=I[i*szp+delt:(i+1)*szp+delt,j*szp+delt:(j+1)*szp+delt]
            idx+=1
    return patches    
    
    
def patch2im(patches,imsize,sz,szp):
    delt,n1,n2=im2patch_len(imsize,sz,szp)
    I=np.zeros(imsize)
    idx=0
    for i in range(n1):
        for j in range(n2):
            I[i*szp+delt:(i+1)*szp+delt,j*szp+delt:(j+1)*szp+delt]=patches[idx,:,:,:]
            idx+=1
    return I    
    
    
def im2patch_len(imzize,sz,szp):
    delt=int(np.ceil((sz-szp)*1.0/2))
    n1=int(np.floor((imzize[0]-2*delt)*1.0/szp))
    n2=int(np.floor((imzize[1]-2*delt)*1.0/szp))
    return delt,n1,n2


def apply_model(I,model,sz,szp):
    Ipatch=im2patch(I,sz,szp)
    Ipred=model.predict(Ipatch)
    return patch2im(Ipred,I.shape,sz,szp)
    
    
    
