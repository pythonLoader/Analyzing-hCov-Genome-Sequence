
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,Conv1D,MaxPooling1D,Activation,Flatten,Bidirectional
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import np_utils
from keras.optimizers import Adam,SGD,RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score

import pandas as pd 
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
import collections
import os
from random import randrange
from numpy import save
from numpy import load

##GLOBAL DEFINITION####

INPUT_FOLDER = "data/"
OUTPUT_FOLDER = "results/"

direc = INPUT_FOLDER+"All_Countries_Representative_Seq"
# cluster_no = ["0","1","2"]#,"3"]
Fast_Vector_direc = INPUT_FOLDER+"All_Countries_NFV_Vects"

Other_Info_File = INPUT_FOLDER+"All_Train_Test_Gisaid.csv"
final_cols = ["Accession ID","Virus name","Location","Collection date","Death","Fast Vector","Sequence"]
all_cluster_direc = INPUT_FOLDER+"All_Clusters"
if not os.path.exists(all_cluster_direc):
    os.mkdir(all_cluster_direc)

df_1 = pd.read_csv(INPUT_FOLDER+'First Cases and Probable Sources .csv')
df_1["First Case"] = pd.to_datetime(df_1["First Case"])
date_ = df_1["First Case"]



date_country_map = {}
for date in date_:
    # print(date.date())
    same_date_df = df_1.loc[df_1["First Case"] == date]
    # print(date,same_date_df["Country"].values)
    if date.date() not in date_country_map:
        date_country_map[date.date()] = same_date_df["Country"].values

print(len(date_country_map))
dub = date_country_map
date_country_map = collections.OrderedDict(sorted(dub.items()))

class Seq_Wrapper:
    def __init__(self,seq,use_indicator=0):
        self.seq = seq
        self.indicator = use_indicator
    def __repr__(self):
        return "<Seq:%s, Indicator:%d>" % (self.seq, self.indicator)

    def __str__(self):
        return "<Seq:%s, Indicator:%d>" % (self.seq, self.indicator)
    
    def get_indicator(self):
        return self.indicator
    
    def get_seq(self):
        return self.seq

    def set_indi(self,indicator):
      self.indicator = indicator

##Normal_Version

final_input_map = {}
Other_df = pd.read_csv(Other_Info_File)
for point in date_country_map:
    country_list = date_country_map[point]
    final_input_map[point] = {}
    
    for country in country_list:
        if(country == "USA"):
            continue
        s_name = country
        d_name = country
        if(country == "USA_Georgia"):
            s_name = "USA / Georgia"
        if(country == "Korea" or country == "South Korea"):
            d_name = "South Korea"
        if(country == "England" or country == "United Kingdom"):
            d_name = "United Kingdom"
        
        
        # print(s_name)
        temp_ = Other_df.loc[Other_df["Location"].str.contains(s_name)]
        # print(s_name,len(temp_))
        if(len(temp_) == 0):
            continue
        
     
        if(d_name not in final_input_map[point]):
            final_input_map[point][d_name] = [Seq_Wrapper(t_seq[19570:19575]) for t_seq in temp_["Sequence"]]
            
            
        else:
            final_input_map[point][d_name].extend([Seq_Wrapper(t_seq[19570:19575]) for t_seq in temp_["Sequence"]])

count=0
for point in final_input_map:
    print(point,len(final_input_map[point]))
    if(len(final_input_map[point]) == 0):
        del date_country_map[point] 
    count+=len(final_input_map[point])

print(len(date_country_map),count)

for key in date_country_map:
    key = key.strftime("%d_%m_%Y")
    if not os.path.exists(all_cluster_direc+"/"+key):
        os.mkdir(all_cluster_direc+"/"+key)

def formatter(date_):
    return date_.strftime("%d_%m_%Y")

final_input_map = {}
for point in date_country_map:
    final_input_map[point] = {}

twenty_seq_df = []
# values = date_country_map.values()
# keys = date_country_map.keys()
# print(values)

def info_sequencer(cluster_file,num):
    print("Working with cluster", cluster_file)
    c_fl = open(cluster_file,"r")
    id_list = c_fl.read().split("\n")
    Fast_Vector_File = Fast_Vector_direc + "/" + num + "_Novel_Fast_vector.csv"
    Fast_vector_df = pd.read_csv(Fast_Vector_File)
    Other_Info_df = pd.read_csv(Other_Info_File)
    final_df_1 = Fast_vector_df.loc[Fast_vector_df["Accession ID"].isin(id_list)]
    final_df_2 = Other_Info_df.loc[Other_Info_df["Accession ID"].isin(id_list)]
    print(final_df_1.shape)
    print(final_df_2.shape)
    final_df = pd.merge(final_df_1, final_df_2, on='Accession ID')
    final_df = final_df[final_cols]

    print(final_df.columns,final_df.shape)

    # direc_point = date_country_map.keys()[date_country_map.values().index(num)]
    for point,maps in date_country_map.items():
        if(num in maps):
            direc_point = formatter(point)
            # sequences = final_df["Sequence"]
            final_input_map[point][num] = [Seq_Wrapper(t_seq[19570:19575]) for t_seq in final_df["Sequence"]]
    twenty_seq_df.append(final_df)
    final_df_name = num+ "_representative_seq" +".csv"
    final_df.to_csv(all_cluster_direc +"/"+direc_point+"/"+final_df_name)

file_list = os.listdir(direc)
count=0
for cluster_file in file_list:
    if(cluster_file.endswith("txt")):
        if(cluster_file.startswith("USA")):
            bleh = cluster_file.split("_")
            num = bleh[0] + "_" + bleh[1]
        else:
            num = cluster_file.split("_")[0]

        # num = cluster_file.split("_")[0]
        count+=1
        info_sequencer(direc+"/"+cluster_file,num)

    
    # cluster_file = direc +"/cluster_."+num+".txt"
print(count)

for point in final_input_map:
    for small_clust in final_input_map[point]:
        print(point,small_clust,len(final_input_map[point][small_clust]))

dub = final_input_map
final_input_map = collections.OrderedDict(sorted(dub.items()))
# pprint(date_country_map)

def get_cluster_dist(cluster1,cluster2):
    vect1 = cluster1["Fast Vector"]
    # print(vect1[0])
    vect2 = cluster2["Fast Vector"]
    # print(vect2.shape)
    final_vect = np.zeros([vect1.shape[0],vect2.shape[0]])
    # print(final_vect.shape)
    for i in range(vect1.shape[0]):
        for j in range(vect2.shape[0]):
            var_1 = np.asarray(vect1[i].replace("[","").replace("]","").replace(' ','').split(",")).astype(np.float)
            var_2 = np.asarray(vect2[j].replace("[","").replace("]","").replace(' ','').split(",")).astype(np.float)
            # if(i == 0 and j == 0):
            #     print(var_1,type(var_1))
            #     print(var_2,type(var_2))
            final_vect[i][j] = np.linalg.norm((var_1-var_2),ord=2)
    dist = np.sum(final_vect)
    # print(final_vect)
    # print(dist)
    return dist

def file_name_checker(cluster_file):
    if(cluster_file.startswith("USA")):
        bleh = cluster_file.split("_")
        num = bleh[0] + "_" + bleh[1]
    else:
        num = cluster_file.split("_")[0]
    return num

def file_array_formatter(file_arr):
    return [file_name_checker(fl) for fl in file_arr]

keys = list(date_country_map.keys())
edge_map = {}
for idx in range(len(date_country_map)-1):
    point_i = keys[idx]
    point_j = keys[idx+1]
    
    file_list_i = os.listdir(all_cluster_direc+"/"+formatter(point_i))
    file_list_j = os.listdir(all_cluster_direc+"/"+formatter(point_j))

    val_i = file_array_formatter(file_list_i)
    val_j = file_array_formatter(file_list_j)
    edge_map[point_i] = {}
    if(len(val_i) == 0 or len(val_j) == 0):
        continue
    for val in val_i:
        edge_map[point_i][val] = val_j
   
    print(len(val_i),len(val_j))


pprint(edge_map)

pprint(final_input_map)

import datetime


############### generating training set #################################
INF = 99999999999
total_time_series = 300000
# total_time_series = 1
time_series_all = {}
total_unique_seq = 0

# count = 0
# BD_first_date = datetime.date(2020, 3, 9)
# ID_first_date = datetime.date(2020, 1, 31)
# PK_first_date = datetime.date(2020, 1, 30)
# MN_first_date = datetime.date(2020, 3, 24)
for i in range(INF):
    one_time_series = ""
    current_country = "China"
    for date in final_input_map.keys():
        # print(date)
        # if date == BD_first_date:
        #   break
        # select a sequence from that country
        total_seq = len(final_input_map[date][current_country])
        seq_idx = -1
        for i in range(total_seq):
            if final_input_map[date][current_country][i].get_indicator() == 0:
                seq_idx = i
                ###### set indicator 1
                final_input_map[date][current_country][i].set_indi(1)
                break
        if seq_idx == -1:
            seq_idx = randrange(total_seq) 

        sequence = final_input_map[date][current_country][seq_idx].get_seq()
#         count = count+1
        one_time_series = one_time_series + sequence
        
        
        # find next country from edge_map to select seq from
        if date in edge_map.keys():        
            total_next_country = len(edge_map[date][current_country])
            next_country_idx = randrange(total_next_country)
            current_country = edge_map[date][current_country][next_country_idx]

    print(one_time_series)
    if not (one_time_series in time_series_all.keys()):
        time_series_all[one_time_series] = 1
        total_unique_seq = total_unique_seq + 1
        print(total_unique_seq)
    if total_unique_seq == total_time_series:
        print("breaking------")
        break

print(len(time_series_all))
# print(count)

list_time_series_all = list(time_series_all.keys())

with open(OUTPUT_FOLDER+'all_time_series.txt', 'w') as f:
    for item in list_time_series_all:
        f.write("%s\n" % item)

####### run at the time of loading the time series sequences #############
with open('results/all_time_series.txt', 'r') as f:
    list_time_series_all = f.read().splitlines()



######### mapping chars to integers ##############
chars = "ATGC"
char_to_int = dict((c, i) for i, c in enumerate(chars))
# print(char_to_int)

##### normalizing dictionary for better performance #########
for letter in char_to_int.keys():
    char_to_int[letter] = char_to_int[letter]/4.0
    
print(char_to_int)

char_to_int_for_y = dict((c, i) for i, c in enumerate(chars))
print(char_to_int_for_y)


POS_OF_MUTATION = 2

list_time_series_all

################## taking full sequence ###################
n_chars = len(list_time_series_all[0])
print(n_chars)
step = 1

pos = 3 ## if pos =1 then, it predicts for 4th base pair. likewise,for,pos = 2, predicts 3th, for pos=3, predicts 2th, for pos=4 predicts 1th,for pos=5 predicts 0th
input_for_lstm = []
output_for_lstm = []

for time_series in list_time_series_all:   
    dataX = []

    for i in range(0, n_chars-pos, step):
        seq_in = time_series[i]
        dataX.append(char_to_int_for_y[seq_in])

    seq_out = time_series[n_chars-pos]

    one_output = char_to_int_for_y[seq_out]
#     one_output = to_categorical(one_output)
    output_for_lstm.append(one_output)
    
    # dataX = np_utils.to_categorical(dataX)
    input_for_lstm.append(dataX)

print(len(input_for_lstm))
print(len(output_for_lstm))

len(input_for_lstm[0])

input_for_lstm = np.array(input_for_lstm)
output_for_lstm = np.array(output_for_lstm)

output_for_lstm = np_utils.to_categorical(output_for_lstm)

input_for_lstm = np_utils.to_categorical(input_for_lstm)

input_for_lstm[0]

input_for_lstm.shape

output_for_lstm.shape

X_train, X_test, y_train, y_test = train_test_split(
input_for_lstm, output_for_lstm, test_size=0.20, random_state=42)
print(X_train.shape)
print(X_test.shape)

###### saving numpy arrays ###########3
save(OUTPUT_FOLDER+'X_train_pos_'+str(POS_OF_MUTATION)+'.npy', X_train)
save(OUTPUT_FOLDER+'X_test_pos_'+str(POS_OF_MUTATION)+'.npy', X_test)
save(OUTPUT_FOLDER+'y_train_pos_'+str(POS_OF_MUTATION)+'.npy', y_train)
save(OUTPUT_FOLDER+'y_test_pos_'+str(POS_OF_MUTATION)+'.npy', y_test)

########## loading numpy arrays #############3
X_train = np.load('results/X_train_pos_0_BD.npy')
X_test = np.load('results/X_test_pos_0_BD.npy')
y_train = np.load('results/y_train_pos_0_BD.npy')
y_test = np.load('results/y_test_pos_0_BD.npy')

LOSS_FN = 'categorical_crossentropy'
LEARNING_RATE = 0.000001
OUTPUT_DIM_RNN_LAYER = 256
RNN_layer = LSTM
BATCH_SIZE = 64 
N_EPOCHS = 5



def CNN_LSTM(dim1,dim2,output_dim):
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=(dim1,dim2)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(256,return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model

model = CNN_LSTM(X_train.shape[1], X_train.shape[2], y_train.shape[1])
model.summary()

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

model = model_1layer_RNN(RNN_layer, OUTPUT_DIM_RNN_LAYER, X_train.shape[1], X_train.shape[2], y_train.shape[1], LOSS_FN, LEARNING_RATE)
model.summary()

callbacks_list = [
    ModelCheckpoint(
        # filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        filepath='best_model_CNN_LSTM_BD_val_15.h5',
        monitor='val_loss', save_best_only=True),
    # EarlyStopping(monitor='val_loss',mode='min', patience=200)
]

history = model.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=N_EPOCHS,
                      # callbacks=[reduce_lr],
                      callbacks=callbacks_list,
                      validation_split=0.15,
                      verbose=1)
model1 = load_model("best_model_CNN_LSTM_BD_val_15.h5")
model1.save(OUTPUT_FOLDER+'CNN_BI_LSTM_pos_'+str(POS_OF_MUTATION)+'_val_15.h5')


y_pred = model1.predict(X_test, verbose=1)

from tensorflow.keras import backend as K
import tensorflow.keras

#### manual accuracy #############
count = 0
count1 = 0
for i  in range(y_test.shape[0]):
  if (y_test[i] == np.round_(y_pred[i])).all():
    count = count+1
    print("true:",y_test[i],"  pred:",np.round_(y_pred[i]))
  else:
    # print("true:",y_test[i],"  pred:",np.round_(y_pred[i]))
    count1 = count1+1
print(count)
print(count1)

print(count/(count+count1))
