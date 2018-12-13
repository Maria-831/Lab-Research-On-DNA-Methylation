import numpy as np
np.random.seed(3)
import pybedtools
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout,BatchNormalization,Activation
from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D,GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Dropout
from keras.optimizers import Adam,RMSprop
from keras import regularizers as kr
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils, to_categorical
# custom R2-score metrics for keras backend
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

import os
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn import linear_model
from sklearn.svm import LinearSVC


#parameter setting...
cell_type = 5
apply_sample_weight = False
target_length = 600
filter_length = 5


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
    n = 0
    for index in range(len(DNA_seq_list)//2):
        DNA_seq.append(DNA_seq_list[index*2 + 1].upper())
        if len(DNA_seq_list[index*2 + 1]) < m:
            m = len(DNA_seq_list[index*2 + 1])
        if len(DNA_seq_list[index*2 + 1]) > n:
            n = len(DNA_seq_list[index*2 + 1])
    print('The shortest length of DNA sequence is {0}bp'.format(m))
    print('The longest length of DNA sequence is {0}bp'.format(n))
    print('Total Number of input sequence is {0}'.format(len(DNA_seq)))
    return DNA_seq,n,m

def extend_Data(targetLength,dnaSeqList):
    newDNAList = []
    for seq in dnaSeqList:
        if len(seq) < targetLength:
            diff = targetLength - len(seq)
            if diff % 2 == 0:
                seq += 'N' * (diff//2)
                seq = 'N' * (diff//2) + seq
            if diff % 2 ==1:
                seq += 'N' *(diff//2)
                seq = 'N' * (diff//2 + 1) + seq
        newDNAList.append(seq)
    return newDNAList

def chop_Data(targetLength,dnaSeqList):
    #chop DNA sequences to have same length
    Uni_DNA = []
    for s in dnaSeqList:
        if len(s) < targetLength:
            print('Exceptions!')
        diff = len(s) - targetLength
        if diff % 2 == 0:
            side = diff // 2
            Uni_DNA.append(s[side:-side])
        else:
            right = diff // 2
            left = diff// 2 + 1
            Uni_DNA.append(s[left:-right])
    return Uni_DNA


def preprocess_data(DNA_seq):

    train_size = len(DNA_seq)

    #One hot encoding 
    DNA = []
    for u in DNA_seq:
        sequence_vector = []
        for c in u:
            if c == 'A':
                sequence_vector.append([1,0,0,0])
            elif c == 'C':
                sequence_vector.append([0,1,0,0])
            elif c == 'G':
                sequence_vector.append([0,0,1,0])
            else:
                sequence_vector.append([0,0,0,1])
        DNA.append(np.array(sequence_vector))
    DNA = np.array(DNA)
    print(DNA.shape)
    return DNA

def Formalize_Data(DNA_seq, methylation_file, target_length, cell_type):
    #Read Methylation level
    labels = list(pd.read_csv(methylation_file,header = None)[cell_type])
    train_labels = np.array(labels)
    training_seq_shape = (len(DNA_seq),target_length,4)
    train_data = DNA_seq.reshape(training_seq_shape)
    return train_data,train_labels


bed_file_path = '/home/h5li/methylation_DMR/data/DMR_coordinates_extended_b500.bed'
fasta_file_path = '/home/h5li/methylation_DMR/data/mm10.fasta'
methylation_file_path = '../../data/Mouse_DMRs_methylation_level.csv'
total_counts_file_path ='../../data/Mouse_DMRs_counts_total.csv'
methy_counts_file_path = '../../data/Mouse_DMRs_counts_methylated.csv'
    
DNA_seq,long_length,short_length = read_data(bed_file_path, fasta_file_path)   

DNA_seq = chop_Data(target_length,DNA_seq)
counts = pd.read_csv('Kmers6_counts_600bp.csv')


def CompareCNN_LASSO( groundtruth, DNA_seq, counts):
    print('')    
    print('Ground Truth Motif is: ',groundtruth)
    methylation_level = []
    DNA_len100 = []
    m1 = 0
    m0 = 0
    for n in range(len(DNA_seq)):
        hasKmers = False
        for t in groundtruth:
            if t in DNA_seq[n]:
                methylation_level.append(1)
                DNA_len100.append(DNA_seq[n])
                hasKmers = True
                break;
        if not hasKmers:
            methylation_level.append(0)
            DNA_len100.append(DNA_seq[n])
    print('Ratio of this motif: ',sum(methylation_level)/len(methylation_level))
    DNA = preprocess_data(DNA_len100)
    train_data,train_labels = Formalize_Data(DNA, methylation_file_path, target_length, cell_type)
    #train_labels = train_labels - np.full((train_labels.shape),np.mean(train_labels))
    #train_labels = to_categorical(np.array(methylation_level))
    train_labels = np.array(methylation_level)
    init = initializers.RandomNormal(mean=0, stddev=0.5, seed=None)
    k_r = kr.l2(1e-6)
    
    print('Strat Training on CNN')
    nfilt = 1
    filtlength = 6
    num_filters = 1
    maxPsize = 100
    seqlen = target_length;
    model = Sequential()
    model.add(Conv1D(filters=num_filters, kernel_size=filtlength,kernel_initializer = 'ones',padding = 'same',
                 input_shape=(seqlen,4), activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, kernel_initializer= 'ones' ,activation='sigmoid'))
    model.compile(optimizer= Adam(lr = 0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=10,mode = 'min')]
    history = model.fit(train_data, train_labels, epochs=500, callbacks = callbacks,
                    validation_split = 0.25,shuffle = False,
                        batch_size=100,verbose=0)
    print("\t CNN Train Accuracy: ", history.history['acc'][-1])
    print("\t CNN Test Accuracy: ", history.history['val_acc'][-1])


    #RUN LASSO
    x = counts.as_matrix()
    y = np.array(methylation_level)
    data = np.concatenate((x, y.reshape(-1,1)), axis=1)
    np.random.shuffle(data)
    train = data[:48000]
    test = data[48000:]
    train_features = train[:,:-1]
    train_methy_levels = train[:,-1]
    test_features = test[:,:-1]
    test_methy_levels = test[:,-1]
    
    C_list = [10**-3,10**-1,1]
    print("")
    print('Training for SVM')
    for c in C_list:
        clf = LinearSVC(penalty='l2',loss = 'hinge', C = c)
        clf.fit(train_features,train_methy_levels)
        a_score = clf.score(test_features,test_methy_levels)
        print('\tParameter: ',c,'Test_accuracy: ',a_score,'Train Accuracy: ',clf.score(train_features,train_methy_levels))
        
    print("========================================================")
    
    
translate = {0:'A',1:'C',2:'G',3:'T'}

for i in range(10):
    motif = ''
    for i in range(6):
        num = random.randint(0,3)
        motif += translate[num]
    CompareCNN_LASSO([motif],DNA_seq,counts)
