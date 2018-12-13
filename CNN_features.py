import numpy as np
from pybedtools import BedTool
import pybedtools
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import regularizers as kr
from tensorflow.python.keras import initializers
from tensorflow.python.keras.callbacks import EarlyStopping
# custom R2-score metrics for keras backend
from keras import backend as K
import matplotlib.pyplot as plt

import os
from sklearn.metrics import r2_score
counts = pd.read_csv('../data/Kmers6_position_counts_extended_DMRs.csv')
CG_attributes = pd.read_csv('../data/CpG_attributes.csv')
counts = pd.concat([counts,CG_attributes],axis = 1)
TF_binding_1 = pd.read_csv('../data/PWM_TFbinding.csv')
TF_binding_2 = pd.read_csv('../data/PWM_TFbinding_rest_600.csv')
TF_binding = pd.concat([TF_binding_1,TF_binding_2],axis = 1)
counts = pd.concat([counts,TF_binding],axis = 1)

labels = pd.read_csv('../data/Mouse_DMRs_methylation_level.csv',header = None)

labels = pd.read_csv('../data/Mouse_DMRs_methylation_level.csv',header = None)

labels = np.array(labels[5])
print(counts.shape,labels.shape)

model = Sequential()
model.add(Dense(counts.shape[1]))
model.add(Dense(1))

def mse_keras(y_true, y_pred):
    SS_res =  K.sum( K.square( y_true - y_pred ) ) 
    SS_tot = K.sum( K.square( y_true - K.mean( y_true ) ) ) 
    return ( SS_res/SS_tot)

def R2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot) )
callbacks = [EarlyStopping(monitor='val_R2_score', patience=10,mode = 'max')]
model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['accuracy',R2_score])

history = model.fit(counts, labels, epochs=500, validation_split = 0.2,shuffle = True,callbacks=callbacks,
                        batch_size=64,verbose=1)


