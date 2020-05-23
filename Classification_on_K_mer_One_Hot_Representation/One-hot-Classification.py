#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from itertools import permutations,combinations_with_replacement


import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Input, concatenate
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split




df_test=pd.read_csv('Test_gisaid_LS.csv')
df_train=pd.read_csv('Train_gisaid_LS.csv')





df_test.head()





train_sequences=list(df_train["Sequence"])
y_indicator=list(df_train["Indicator"])
print(len(train_sequences))





K_MER_LENGTH = 3 
NUMBER_OF_K_MERS=np.power(4,K_MER_LENGTH)
print(NUMBER_OF_K_MERS)




def get_motifs(length,seq):
    (d,index) = ({}, 0)
    permitted_list = ['A','C','G','T']
    seq_k_mer_list = []
    for i in range(0, len(seq)-length+1):
        flag=0
        word = seq[i:i+length]
        for letter in word:
            if letter not in permitted_list:
                flag=1
        if(flag == 1):
            continue
        seq_k_mer_list.append(word)

    return seq_k_mer_list



def get_motifs_V2(length):
    lst = ['A','C','G','T','N']
    els =  list(itertools.product(lst,repeat = length))
    els = [' '.join(tups) for tups in els]
    return els 


def get_all_combs(seq,power):
  ret_comb=[]
  c1=combinations_with_replacement(seq,power)
  for c in c1:
    p=permutations(c,power)
    new_list=[]
    lst=list(p)
    for tup in lst:
      str_new =  ''.join(tup) 
      new_list.append(str_new)
    unique_list=np.unique(new_list)
    for u in unique_list:
      ret_comb.append(u)
  return ret_comb



length_max=0
for i in range(0,len(train_sequences)):
  k_mer_list=get_motifs(K_MER_LENGTH,train_sequences[i])
  if len(k_mer_list)>length_max:
    length_max=len(k_mer_list)
print(length_max)




MAX_LEN=length_max



all_combination_list=get_all_combs('ATCG',K_MER_LENGTH)
all_comb_dict={}
for i in range(0,len(all_combination_list)):
  all_comb_dict[all_combination_list[i]]=i
print(all_comb_dict)




print(len(all_combination_list))




import pandas as pd
def one_hot_encode(d): #Pass the motif dictionary
    S = pd.Series({'A':list(d.keys())})
    # print(S)
    one_hot = pd.get_dummies(S['A'])
    print(type(one_hot))
    return one_hot




one_hot_df=one_hot_encode(all_comb_dict)





def get_max_len(all_seq_list):
  max_len=0
  for i in range(0,len(all_seq_list)):
    lst=all_seq_list[i]
    if len(lst)>max_len:
      max_len=len(lst)
  return max_len




def sequence_generator(all_seq_list,max_len, all_comb_dict,bs, y_vals):
  num=0
  while True:
    sequences_2D_Mats=[]
    indicators=[]
    while len(indicators)<bs:
      sequence=all_seq_list[num]      
      sequence_k_mers=get_motifs(K_MER_LENGTH,sequence)
      #print(len(sequence_k_mers))
      Mat_2D=np.zeros((NUMBER_OF_K_MERS,max_len))
      for col_val in range(0,len(sequence_k_mers)):
        k_mer=sequence_k_mers[col_val]
        row_value=all_comb_dict[k_mer]
        Mat_2D[row_value,col_val]=1
      sequences_2D_Mats.append(Mat_2D)
      indicators.append(y_vals[num])
      num+=1
      if num==len(all_seq_list):
        print(num)
        num=0
    yield (np.array(sequences_2D_Mats),indicators)
    
      




train_sequences=np.array(train_sequences)
y_indicator=np.array(y_indicator)





X_train, X_val, y_train, y_val = train_test_split(
train_sequences, y_indicator, test_size=0.15, random_state=42)
print(X_train.shape)
print(X_val.shape)




NUM_EPOCHS = 150
BS = 10
Total_Epochs_Running=150


LOSS_FN = 'binary_crossentropy'  
LEARNING_RATE = 0.000001   # 0.0001, 0.00005, 0.00003



num_train_seq=len(X_train)
num_valid_seq=len(X_val)





max_len=MAX_LEN
print(type(all_comb_dict))
trainGen = sequence_generator(X_train,max_len, all_comb_dict,BS,y_train)
validationGen = sequence_generator(X_val,max_len, all_comb_dict,BS,y_val)





def AlexNet(d1,d2):
  model = Sequential()

  # 1st Convolutional Layer
  model.add(Conv1D(filters=96, input_shape=(d1,d2), kernel_size=11, strides=4, padding="valid", activation = "relu"))

  # Max Pooling
  model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))

  # 2nd Convolutional Layer
  model.add(Conv1D(filters=256, kernel_size=5, strides=1, padding="same", activation = "relu"))

  # Max Pooling
  model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))

  # 3rd Convolutional Layer
  model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding="same", activation = "relu"))

  # 4th Convolutional Layer
  model.add(Conv1D(filters=384, kernel_size=3, strides=1, padding="same", activation = "relu"))

  # 5th Convolutional Layer
  model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding="same", activation = "relu"))

  # Max Pooling
  model.add(MaxPool1D(pool_size=2, strides=2, padding="valid"))

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
    # 1x1 conv
    conv1 = Conv1D(f1, 1, padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv1D(f2_in, 1, padding='same', activation='relu')(layer_in)
    conv3 = Conv1D(f2_out, 3, padding='same', activation='relu')(conv3)
    # 5x5 conv
    conv5 = Conv1D(f3_in, 1,padding='same', activation='relu')(layer_in)
    conv5 = Conv1D(f3_out, 5, padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = MaxPool1D(3, strides=1, padding='same')(layer_in)
    pool = Conv1D(f4_out, 1, padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)

    return layer_out
 
def InceptionNet(dim1,dim2):
    # define model input
    visible = Input(shape=(dim1,dim2))
    # add inception block 1
    layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
    # add inception block 1
    layer = inception_module(layer, 128, 128, 192, 32, 96, 64)
    # create model

    x = Flatten()(layer)
    x = Dense(50, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=visible, outputs=predictions)
    model.compile(loss=LOSS_FN, optimizer=Adam(lr=LEARNING_RATE, amsgrad=True), metrics=['accuracy'])
    # summarize model
    model.summary()
    return model






model=AlexNet(NUMBER_OF_K_MERS,MAX_LEN)
model_name="AlexNet"

model_name="InceptionNet"
model=InceptionNet(NUMBER_OF_K_MERS,MAX_LEN)





callbacks_list = [
    ModelCheckpoint(
        # filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        filepath='best_model_'+str(K_MER_LENGTH)+'_'+model_name+'_BS_'+str(BS)+'_E_'+str(Total_Epochs_Running)+'.h5',
        monitor='val_loss', save_best_only=True),
    # EarlyStopping(monitor='val_loss',mode='min', patience=200)
]





H = model.fit_generator(
    trainGen,
    steps_per_epoch=num_train_seq // BS,
    validation_data=validationGen,
    validation_steps=num_valid_seq // BS,
    epochs=NUM_EPOCHS,
 callbacks=callbacks_list)



model.save(model_name+"_K_"+str(K_MER_LENGTH)+'_BS_'+str(BS)+'_E_'+str(Total_Epochs_Running)+'.h5')

