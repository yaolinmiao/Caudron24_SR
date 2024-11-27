#!/usr/bin/env python
# coding: utf-8

import glob
import obspy
from obspy import UTCDateTime as UTC
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.tf_misfit import cwt
import os
import sys

from sklearn.decomposition import PCA 
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from tensorflow.python import keras
from tensorflow.keras.optimizers import Adam,SGD
from keras.layers import Layer, InputSpec
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout, Reshape 
from keras.layers import Bidirectional, BatchNormalization, ZeroPadding1D, Conv2DTranspose
from keras.models import Model,load_model
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.initializers import VarianceScaling


mseeds=np.sort(glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/for_clustering/ch144/*.mseed'))

inp = Input(shape=(128, 1600, 1))  
e = Conv2D(4, (3, 3), activation='tanh', padding='same')(inp)
e = MaxPooling2D((2, 2), padding='same')(e)
e = Conv2D(2, (3, 3), activation='tanh', padding='same')(e)
e = MaxPooling2D((2, 2), padding='same')(e)
e = Conv2D(2, (3, 3), activation='tanh', padding='same')(e)
e = MaxPooling2D((2, 2), padding='same')(e)
e = Conv2D(2, (3, 3), activation='tanh', padding='same')(e)
e = MaxPooling2D((2, 2), padding='same')(e)
e = Conv2D(1, (3, 3), activation='tanh', padding='same')(e)
e = MaxPooling2D((2, 2), padding='same')(e)

shape_before_flattening = K.int_shape(e)
encoded = Flatten()(e)
d = Reshape(shape_before_flattening[1:])(encoded)

d = Conv2D(1, (3, 3), activation='tanh', padding='same')(d)
d = UpSampling2D((2, 2))(d)
d = Conv2D(2, (3, 3), activation='tanh', padding='same')(d)
d = UpSampling2D((2, 2))(d)
d = Conv2D(2, (3, 3), activation='tanh', padding='same')(d)
d = UpSampling2D((2, 2))(d)
d = Conv2D(2, (3, 3), activation='tanh', padding='same')(d)
d = UpSampling2D((2, 2))(d)
d = Conv2D(4, (3, 3), activation='tanh', padding='same')(d)
d = UpSampling2D((2, 2))(d)
decoded = Conv2D(1, (3, 3), padding='same')(d)

autoencoder = Model(inputs=inp, outputs=decoded, name='autoencoder')
encoder = Model(inputs=inp, outputs=encoded, name='encoder')

training_index=int(sys.argv[1])

cwtmat=np.load('/nfs/turbo/lsa-zspica/work/yaolinm/allcwtmat.npy')

autoencoder.compile(optimizer='adam', loss='mse')
overfitCallback = EarlyStopping(monitor='loss', min_delta=0.00005, patience=5)
autoencoder.fit(cwtmat, cwtmat, batch_size=32, epochs=1000, callbacks=[overfitCallback])

def cal_acc(training_labels,pred_labels):
    
    from itertools import permutations
    perm=list(permutations([0,1,2,3]))
    cs=[]
    for p in perm:
        c=0
        for i in range(len(pred_labels)):
            if pred_labels[i]==0:
                if int(training_labels[0,i])==p[0]:
                    c+=1
            elif pred_labels[i]==1:
                if int(training_labels[0,i])==p[1]:
                    c+=1
            elif pred_labels[i]==2:
                if int(training_labels[0,i])==p[2]:
                    c+=1
            elif pred_labels[i]==3:
                if int(training_labels[0,i])==p[3]:
                    c+=1
        cs.append(c)
        
    return np.max(cs)/200
    
kmeans = KMeans(n_clusters=4, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(cwtmat[:1000,:,:,:]))

training_labels=np.load('training_labels.npy')
pred_labels=[]
for i in range(len(training_labels[1,:])):
    index=int(training_labels[1,i])
    pred_labels.append(y_pred[index])
    
accuracy1=cal_acc(training_labels,pred_labels)

kmeans = KMeans(n_clusters=4, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(cwtmat[:,:,:,:]))

training_labels=np.load('training_labels.npy')
pred_labels=[]
for i in range(len(training_labels[1,:])):
    index=int(training_labels[1,i])
    pred_labels.append(y_pred[index])
    
accuracy2=cal_acc(training_labels,pred_labels)    

autoencoder.save('models/repetitive/index'+str(training_index).zfill(3)+'_autoencoder.h5')
encoder.save('models/repetitive/index'+str(training_index).zfill(3)+'_encoder.h5')
f=open('acclog.txt', 'w+')
print('index: '+str(training_index).zfill(3)+', accuracy1'+str(round(accuracy1,1))+', accuracy2'+str(round(accuracy2,1)))
print(str(training_index).zfill(3)+', '+str(round(accuracy1,1))+', '+str(round(accuracy2,1)),file=f)
f.close()




