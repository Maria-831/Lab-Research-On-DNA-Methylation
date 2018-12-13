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
# custom R2-score metrics for keras backend
from tensorflow.python.keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def read_data(bed_file,fasta_file):
    #apply bedtools to read fasta files '/home/h5li/methylation_DMR/data/DMR_coordinates_extended_b500.bed'
    a = pybedtools.example_bedtool( bed_file )
    # '/home/h5li/methylation_DMR/data/mm10.fasta'
    fasta = pybedtools.example_filename( fasta_file )
    a = a.sequence(fi=fasta)
    seq = open(a.seqfn).read()
    #read and extract DNA sequences 
    DNA_seq_list = seq.split('\n')
    DNA_seq_list.pop()
    DNA_seq = []
    m = 10000
    for index in range(len(DNA_seq_list)//2):
        DNA_seq.append(DNA_seq_list[index*2 + 1].upper())
        if len(DNA_seq_list[index*2 + 1]) < m:
            m = len(DNA_seq_list[index*2 + 1])
    print('The shortest length of DNA sequence is {0}bp'.format(m))
    return DNA_seq

#below are helper methods
def data_aug(seq):
    new_seq = []
    for i in range(len(seq)):
        l = seq[i]
        if l == 'A':
            new_seq.append( 'T' )
        elif l == 'C':
            new_seq.append( 'G' )
        elif l == 'G':
            new_seq.append( 'C' )
        else:
            new_seq.append( 'A' )
    return new_seq

def data_rev(seq):
    new_seq = [None] * len(seq)
    for i in range(len(seq)):
        new_seq[-i] = seq[i]
    return new_seq      

def mse_keras(y_true, y_pred):
    SS_res =  K.sum( K.square( y_true - y_pred ) ) 
    SS_tot = K.sum( K.square( y_true - K.mean( y_true ) ) ) 
    return ( SS_res/SS_tot)

def R2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot) )

def preprocess_data(DNA_seq, target_length,data_aug = False):
    #Choose an optimal length
    target_length = target_length

    train_size = len(DNA_seq)

    #chop DNA sequences to have same length
    Uni_DNA = []
    for s in DNA_seq:
        if len(s) < target_length:
            print('Exceptions!')
        diff = len(s) - target_length
        if diff % 2 == 0:
            side = diff // 2
            Uni_DNA.append(s[side:-side])
        else:
            right = diff // 2
            left = diff// 2 + 1
            Uni_DNA.append(s[left:-right])
    
    if data_aug:
        seq = Uni_DNA
        #Data Augmentation
        new_data = []
        for u in seq:
            new_data.append(data_aug(u))
        seq = seq + new_data
    
        new_data = []
        for u in seq:
            new_data.append(data_rev(u))
        Uni_DNA = seq + new_data

    #One hot encoding 
    DNA = []
    for u in Uni_DNA:
        sequence_vector = []
        for mode in ['A','C','G','T']:
            a = []
            for index in range(len(u)):
                if u[index] == mode:
                    a.append(float(1))
                else:
                    a.append(float(0))
            sequence_vector.append(a)
        DNA.append(np.array(sequence_vector))
    DNA = np.array(DNA)
    print(DNA.shape)
    return DNA

def Formalize_Data(DNA_seq, methylation_file, target_length, cell_type):
    #Read Methylation level
    labels = list(pd.read_csv(methylation_file,header = None)[cell_type])
    train_labels = np.array(labels)
    training_image_shape = (len(DNA_seq), 4, target_length)
    train_data = DNA_seq.reshape(training_image_shape)
    return train_data,train_labels

def weight(index):
    if labels[index] == 0:
        weight = - train_methy[i] * np.log(1e-6) - train_unmethy[i] * np.log( 1 - 1e-6)
        return weight
    elif labels[index] == 1:
        weight = - train_methy[i] * np.log(1 - 1e-6) - train_unmethy[i] * np.log( 1e-6 )
        return weight
    else:
        return - train_methy[i] * np.log(labels[i]) - train_unmethy[i] * np.log( 1 - labels[i])


def Generate_Sample_Weight(total_counts, methy_counts,cell_type,data_aug = False):
    #read in total counts
    total = pd.read_csv(total_counts,header = None)[cell_type].as_matrix().astype('float32')
    methy = pd.read_csv(methy_counts,header = None)[cell_type].as_matrix().astype('float32')
    unmethy = total - methy
    train_methy = methy
    train_unmethy = unmethy
    
    
    #generate sample weight
    sample_weight = []
    for i in range(len(train_methy)):
        sample_weight.append(weight(i))
    if data_aug:
        sample_weight = sample_weight*4
    sample_weight = np.array(sample_weight)
    print(sample_weight.shape)
    return sample_weight

def construct_CNN(target_length,numConv,kernel_num,kernel_size,dropout,maxpool = False, 
                  maxpool_size=1 , add_dense_layer = False, dense_unit = None,normalization = False):
    # create model
    model=Sequential()
    model.add(Dropout(0.2))
    
    if numConv != len(kernel_size) and len(kernel_num) != len(kernel_size):
        print('Incompatible number of kernel sizes with number of Conv layer!')
        print('Incompatible number of filters with number of Conv layer!')
    
    #Construct Convolutional Layers
    for n in range(numConv):
        model.add(Conv1D(kernel_num[n], kernel_size = kernel_size[n],padding = 'same', 
                         input_shape = (4, target_length/(maxpool_size ** n)), activation = 'relu'))
        model.add(Dropout(dropout))
        
        if maxpool:
            model.add(MaxPooling1D(pool_size = maxpool_size, padding='same'))
            model.add(Dropout(dropout))
            
        if normalization:
            model.add(BatchNormalization())
    
    # Flatten the network
    model.add(Flatten())
    model.add(Dropout(dropout))
    
    #Construct Dense Layer
    if add_dense_layer:
        for n in range(len(dense_unit)):
            model.add(Dense(dense_unit[n],activation = 'relu'))
            model.add(Dropout(0.2))
    
    model.add(Dense(1))
    return model

def train_CNN(model, data, labels, CNN_param,sample_weight = None,shuffle = True):
    model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['accuracy',R2_score])
    if sample_weight is not None:
        history = model.fit(data, labels, epochs=500, 
                    validation_split = 0.2,shuffle = shuffle,
                    batch_size=CNN_param['batch_size'],sample_weight = sample_weight,verbose=1)
    else:
        history = model.fit(data, labels, epochs=500, 
                    validation_split = 0.2,shuffle = shuffle,
                    batch_size=CNN_param['batch_size'],verbose=1)
    
    # summarize history for loss
    plt.plot(history.history['R2_score'])
    plt.plot(history.history['val_R2_score'])
    plt.title('model R2_score')
    plt.ylabel('R2_score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    path = 'Test'
    plt.savefig(str(path)+'/'+str(CNN_param['numConv']) +'Conv_'+ str(CNN_param['kernel_num'])+ 'kernel_num_' +
                str(CNN_param['kernel_size']) + 'kernel_size' + str(CNN_param['dropout']) + 'dropout_'
                + str(CNN_param['maxpool'])+ str(CNN_param['maxpool_size']) + 'pool_'+ 
                str(CNN_param['dense_unit']) + 'dense_'+ '.png')
    
    
    
def main(target_length,cell_type,apply_data_aug,apply_sample_weight,CNN_param):
    bed_file_path = '/home/h5li/methylation_DMR/data/DMR_coordinates_extended_b500.bed'
    fasta_file_path = '/home/h5li/methylation_DMR/data/mm10.fasta'
    methylation_file_path = '../data/Mouse_DMRs_methylation_level.csv'
    total_counts_file_path ='../data/Mouse_DMRs_counts_total.csv'
    methy_counts_file_path = '../data/Mouse_DMRs_counts_methylated.csv'
    
    target_length = target_length
    cell_type = cell_type
    apply_sample_weight = False
    
    DNA_seq = read_data(bed_file_path, fasta_file_path)
    
    DNA = preprocess_data(DNA_seq, target_length,data_aug = apply_data_aug)
    
    train_data,train_labels = Formalize_Data(DNA, methylation_file_path, target_length, cell_type)
    
    if apply_sample_weight:
        samp_weight = Generate_Sample_Weight(total_counts_file_path,methy_counts_file_path,
                                               cell_type,data_aug = apply_data_aug)
    else:
        samp_weight = None
        
    CNN = construct_CNN(target_length = target_length,numConv = CNN_param['numConv'],
                        kernel_num = CNN_param['kernel_num'],kernel_size = CNN_param['kernel_size'],
                        dropout = CNN_param['dropout'],maxpool = CNN_param['maxpool'], 
                  maxpool_size=CNN_param['maxpool_size'], add_dense_layer = CNN_param['add_dense_layer'], 
                        dense_unit = CNN_param['dense_unit'],normalization = CNN_param['normalization'])
    train_CNN(CNN,train_data,train_labels,CNN_param,sample_weight = samp_weight)
    
if __name__ == "__main__":
    
    CNN1 = {'numConv': 1, 'kernel_num': [10], 'kernel_size': [8], 'dropout': 0.2, 'maxpool': False,
           'maxpool_size':2, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':1000}
    main(600,5,False,False,CNN1)
    
    CNN1 = {'numConv': 1, 'kernel_num': [10], 'kernel_size': [8], 'dropout': 0.2, 'maxpool': False,
           'maxpool_size':2, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN1)
    
    CNN1 = {'numConv': 1, 'kernel_num': [20], 'kernel_size': [11], 'dropout': 0.2, 'maxpool': False,
           'maxpool_size':2, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN1)
    
    CNN4 = {'numConv': 1, 'kernel_num': [30], 'kernel_size': [11], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':4, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN4)
    
    CNN2 = {'numConv': 1, 'kernel_num': [40], 'kernel_size': [8], 'dropout': 0.2, 'maxpool': False,
           'maxpool_size':2, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN2)
    
    CNN4 = {'numConv': 1, 'kernel_num': [30], 'kernel_size': [11], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':4, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(1024,5,False,False,CNN4)
    
    CNN3 = {'numConv': 1, 'kernel_num': [30], 'kernel_size': [11], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':2, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN3)
    
    CNN4 = {'numConv': 1, 'kernel_num': [30], 'kernel_size': [11], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':4, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN4)
    
    CNN5 = {'numConv': 2, 'kernel_num': [30,20], 'kernel_size': [11,6], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':2, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN5)
    
    CNN6 = {'numConv': 2, 'kernel_num': [30,20], 'kernel_size': [11,6], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':4, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':2000}
    main(600,5,False,False,CNN6)
    
    CNN6 = {'numConv': 2, 'kernel_num': [30,20], 'kernel_size': [11,6], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':4, 'add_dense_layer': False, 'dense_unit':[],'normalization':True,'batch_size':2000}
    main(600,5,False,False,CNN6)
    
    CNN6 = {'numConv': 2, 'kernel_num': [30,20], 'kernel_size': [11,6], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':4, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':4000}
    main(600,5,False,False,CNN6)
    
    CNN6 = {'numConv': 2, 'kernel_num': [30,20], 'kernel_size': [11,6], 'dropout': 0.2, 'maxpool': True,
           'maxpool_size':4, 'add_dense_layer': False, 'dense_unit':[],'normalization':False,'batch_size':4000}
    main(600,5,False,False,CNN6)
