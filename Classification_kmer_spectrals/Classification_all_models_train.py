import os
import argparse
import time
import datetime
import pandas as pd
import random
import math
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Flatten,Input,LeakyReLU,BatchNormalization,Reshape,Dropout,Activation,Conv1D,MaxPool1D,concatenate
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import itertools
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix,roc_auc_score,accuracy_score

######### global defination ###############

LOSS_FN = 'binary_crossentropy'  
LEARNING_RATE = 0.000001  
n_epochs = 1000
BATCH_SIZE =32

OUTPUT_FOLDER = "new_models"
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)


def CNN(dim1,dim2):
  model = Sequential()
  model.add(Conv1D(filters=10, kernel_size=5, strides=1, padding="same",activation='relu', input_shape=(dim1,dim2)))

  model.add(MaxPool1D(pool_size=2))
  model.add(BatchNormalization(momentum=0.9))

  model.add(Conv1D(filters=20, kernel_size=5, strides=1, padding="same",activation='relu'))

  model.add(MaxPool1D(pool_size=2))
  model.add(BatchNormalization(momentum=0.9))

  model.add(Conv1D(filters=30, kernel_size=5))
  model.add(MaxPool1D(pool_size=2))
  model.add(BatchNormalization(momentum=0.9))

  model.add(Flatten())
  
  model.add(Dense(500))
  model.add(Dropout(0.5))

  model.add(Dense(1,activation='sigmoid'))

  model.compile(loss=LOSS_FN, optimizer=Adam(lr=LEARNING_RATE, amsgrad=True), metrics=['accuracy'])
  model.summary()

  return model

def AlexNet(d1,d2):
  model = Sequential()

  # 1st Convolutional Layer
  model.add(Conv1D(filters=96, input_shape=(d1,d2), kernel_size=11, strides=3, padding="valid", activation = "relu"))

  # Max Pooling
  model.add(MaxPool1D(pool_size=3, strides=2, padding="valid"))

  # 2nd Convolutional Layer
  model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding="same", activation = "relu"))

  # Max Pooling
  model.add(MaxPool1D(pool_size=3, strides=2, padding="valid"))

  # 3rd Convolutional Layer
  model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding="same", activation = "relu"))

  # 4th Convolutional Layer
  model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding="same", activation = "relu"))

  # 5th Convolutional Layer
  model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding="same", activation = "relu"))

  # Max Pooling
  model.add(MaxPool1D(pool_size=3, strides=2, padding="valid"))

  # Passing it to a Fully Connected layer
  model.add(Flatten())
  # 1st Fully Connected Layer
  model.add(Dense(units = 9216, activation = "relu"))

  # 2nd Fully Connected Layer
  model.add(Dense(units = 4096, activation = "relu"))

  # 3rd Fully Connected Layer
  model.add(Dense(4096, activation = "relu"))

  # Output Layer
  model.add(Dense(1, activation = "sigmoid")) 

  model.compile(loss=LOSS_FN, optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
  model.summary()

  return model

def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    conv1 = Conv1D(f1, 1, padding='same', activation='relu')(layer_in)
  
    conv3 = Conv1D(f2_in, 1, padding='same', activation='relu')(layer_in)
    conv3 = Conv1D(f2_out, 3, padding='same', activation='relu')(conv3)

    conv5 = Conv1D(f3_in, 1,padding='same', activation='relu')(layer_in)
    conv5 = Conv1D(f3_out, 5, padding='same', activation='relu')(conv5)

    pool = MaxPool1D(3, strides=1, padding='same')(layer_in)
    pool = Conv1D(f4_out, 1, padding='same', activation='relu')(pool)

    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)

    return layer_out

def InceptionNet(dim1,dim2):
  visible = Input(shape=(dim1,dim2))
  layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
  layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
  # create model
      
  x = Flatten()(layer)
  x = Dense(50, activation='relu')(x)
  predictions = Dense(1, activation='sigmoid')(x)

  model = Model(inputs=visible, outputs=predictions)
  model.compile(loss=LOSS_FN, optimizer=Adam(lr=LEARNING_RATE, amsgrad=True), metrics=['accuracy'])
  model.summary()
  return model

def get_sequences(input_file):
    (seq, sequences_list)=('',[])
    for line in open(input_file):
        if line.startswith('>'):
            sequences_list.append(seq)
            seq = ''
        else:
            seq+= line.rstrip()
    sequences_list.append(seq)
    del sequences_list[0]
    return sequences_list

def get_motifs(length,sequences_list):
    (d,index) = ({}, 0)
    permitted_list = ['A','C','G','T']
    for seq in sequences_list:
        for i in range(0, len(seq)-length+1):
            flag=0
            word = seq[i:i+length]
            for letter in word:
                if letter not in permitted_list:
                    flag=1
            if(flag == 1):
                print("problem")
                continue
            if word not in d:
                d[word] = index
                index += 1 
    return d

def get_motifs_V2(length):
    (d,index) = ({}, 0)
    lst = ['A','C','G','T']
    els =  list(itertools.product(lst,repeat = length))
    els = [''.join(tups) for tups in els]
    for elems in els:
        d[elems] = index
        index+=1
    return d


def one_hot_encode(d):
    S = pd.Series({'A':d.keys()})
    # print(S)
    one_hot = pd.get_dummies(S['A'])
    print(type(one_hot))
    return one_hot


def calculate_occurrences(length, input_file):
    sequences_list = input_file["Sequence"]
    d = get_motifs_V2(length)   
    rows_num = len(sequences_list)
    cols_num = len(d)
    #with open("k-mers.txt","w") as out_file:
     #   pprint(d, stream=out_file)

    data = np.zeros(shape=(rows_num,cols_num))
    y_indicator = np.zeros(shape=(rows_num,1))
    for row_idx, seq in enumerate(sequences_list):
        for i in range(0, len(seq)-length+1):
            word = seq[i:i+length]
            if word in d:
                col_idx = d[word]
                data[row_idx, col_idx] += 1
        y_indicator[row_idx] = input_file["Indicator"][row_idx]
    return (data,y_indicator)

def calculate_frequencies(occurrences_list,seqs_number):
    frequencies_list =[]
    for i in range(0,seqs_number):
        frequencies_list.append(occurrences_list[i,:]/np.sum(occurrences_list[i,:]))

def change_sequence(sequences):
    change_map = {}
    change_map["R"] = ["A","G"]
    change_map["Y"] = ["C","T"]
    change_map["S"] = ["C","G"]
    change_map["W"] = ["A","T"]
    change_map["K"] = ["G","T"]
    change_map["M"] = ["A","C"]
    change_map["B"] = ["C","T","G"]
    change_map["D"] = ["A","T","G"]
    change_map["H"] = ["C","T","A"]    
    change_map["V"] = ["C","A","G"]
    change_map["N"] = ["A","C","T","G"]
    # pprint(change_map)
    keys = change_map.keys()
    min_val = 10000000
    max_val = -1
    for idx,seq in enumerate(sequences):
        before = len(seq)
        seq = seq.replace("-","")
        after = len(seq)

        mut_seq = [c for c in seq]

        # print(len(mut_seq))
        length = len(mut_seq)
       
        for i in range(len(seq)):
            if(seq[i] in keys):
                mut_seq[i] = random.choice(change_map[seq[i]])

        unique = np.unique(mut_seq)
        if(len(unique) != 4):
            print(idx,unique)

        sequences[idx] = ''.join(mut_seq)


    return sequences

def generator(x,y,batch_size):
  index_list = list(range(x.shape[0]))
  random.shuffle(index_list)
  count = 0
  while True:
    x_gen=[]
    y_indicators=[]
    while len(y_indicators)<batch_size:
      x_gen.append(x[index_list[count]])      
      y_indicators.append(y[index_list[count]])
      count+=1

      if count == x.shape[0]:
        count = 0
        random.shuffle(index_list)
    yield (np.array(x_gen),np.array(y_indicators))


def check(sequences):
    min_val = 10000000
    max_val = -1
    for seq in sequences:
        length = len(seq)
        if(length< min_val):
            min_val = length
        if(length > max_val):
            max_val = length
        mut_seq = [c for c in seq]
        unique = np.unique(mut_seq)
        if(len(unique) != 4):
            print(unique)
        # print(len(unique))
    print(min_val,max_val)

def Train(label,modelname,k_length):
    # fetches train csv from Input folder
    # represents train data as kmer spectrals
    # trains and saves model
    
    train_data = pd.read_csv("../Input/Train_labelled_by_"+label+".csv")
    print("preparing input-------------")
    X_spectral_list,y_indicator = calculate_occurrences(k_length,train_data.copy())
    print(X_spectral_list.shape,y_indicator.shape)
    X_spectral_list_reshaped = np.reshape(X_spectral_list,(X_spectral_list.shape[0],X_spectral_list.shape[1],1))
    print(X_spectral_list_reshaped.shape)
    
    if modelname == "CNN":
        model = CNN(X_spectral_list_reshaped.shape[1],X_spectral_list_reshaped.shape[2])
    elif modelname == "AlexNet":
        model = AlexNet(X_spectral_list_reshaped.shape[1],X_spectral_list_reshaped.shape[2])
    elif modelname == "InceptionNet":
        model = InceptionNet(X_spectral_list_reshaped.shape[1],X_spectral_list_reshaped.shape[2])
    
    callbacks_list = [
    ModelCheckpoint(
        filepath=OUTPUT_FOLDER+"/"+'best_model_'+str(k_length)+'_'+modelname+'_val_15.h5',
        monitor='val_loss', save_best_only=True)
    ]
    history = model.fit(X_spectral_list_reshaped,
                      y_indicator,
                      batch_size=BATCH_SIZE,
                      epochs=n_epochs,
                      callbacks=callbacks_list,
                      validation_split=0.15,
                      verbose=1)
    #model.save(modelname+'_k_'+str(k_length)+'_'+'val_15.h5')
    model.save_weights(OUTPUT_FOLDER+"/"+modelname+'_k_'+str(k_length)+'_'+'val_15.h5')
    
    
    ### if training needs larger RAM, use following fit_generator
    
    # X_train, X_val, y_train, y_val = train_test_split(
    # X_spectral_list_reshaped, y_indicator, test_size=0.15, random_state=42)
    # print(X_train.shape)
    # print(X_val.shape)
    # trainGen = generator(X_train,y_train,BATCH_SIZE)
    # validationGen = generator(X_val,y_val,BATCH_SIZE)
    # history = model.fit_generator(
    # 	trainGen,
    # 	steps_per_epoch = X_train.shape[0] // BATCH_SIZE,
    # 	validation_data = validationGen,
    # 	validation_steps = X_val.shape[0] // BATCH_SIZE,
    # 	epochs = n_epochs,
    #   callbacks = callbacks_list)




def main():
    parser = argparse.ArgumentParser(description="classification with input represented as k_mer spectrals.Requires label on which train set is prepared,model name,kmer length.")
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--label','-l',help='Label on which parameter (Death/CFR_confirmed_cases/CFR_Recovery/CFR_Infrastructure)',required=True,type=str)
    required.add_argument('--model','-m',help='Name the model to be trained (CNN/AlexNet/InceptionNet) ', required=True, type=str)
    required.add_argument('--k_length','-kl',help='Define the kmer length (3/5/7)',required=True, type=int)
    
    pars = parser.parse_args()
    label = pars.label
    model = pars.model
    k_length = pars.k_length
    
    Train(label,model,k_length)
    


if __name__ == "__main__":
    main()