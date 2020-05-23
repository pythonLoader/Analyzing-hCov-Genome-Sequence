
import pandas as pd 
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
import collections
import pandas as pd 
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
import collections
import os
from random import randrange
from numpy import save
from numpy import load
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM,Conv1D,MaxPooling1D,Activation,Flatten,Bidirectional
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import np_utils
from keras.optimizers import Adam,SGD,RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score

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

for point in final_input_map:
    for small_clust in final_input_map[point]:
        seq_list = [x.seq for x in final_input_map[point][small_clust]]
        print(point,small_clust,np.unique(seq_list,return_counts=True))
        print()

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

   
    print(len(val_i),len(val_j))
    all_cluster_table = np.full((len(file_list_i),len(file_list_j)),-90)
    # print(all_cluster_table.shape)
    count = 0
    for i,file1 in enumerate(file_list_i):
        region = file_name_checker(file1)
        bleh1 = pd.read_csv(all_cluster_direc+"/"+formatter(point_i)+"/"+file1)
        for j,file2 in enumerate(file_list_j):
            # # if(all_cluster_table)
            # if ((region not in file2) and (all_cluster_table[i][j] == -90)):
            count+=1
            bleh2 = pd.read_csv(all_cluster_direc+"/"+formatter(point_j)+"/" +file2)
            # print(file1,file2)
            all_cluster_table[i][j] = (get_cluster_dist(bleh1,bleh2)/(bleh1.shape[0]*bleh2.shape[0]))

    # # print(point_i,point_j,all_cluster_table.shape)
  
    for kl in val_i:
        edge_map[point_i][kl] = []
    # uc = np.where(all_cluster_table == np.min(all_cluster_table)) #all_cluster_table.argmin()
    # edge_map[point_i][val_i[uc[0][0]]].append(val_j[uc[1][0]])
    # print(uc[0][0],uc[1][0])
    if(len(val_i) < len(val_j)):
        uc = all_cluster_table.argmin(axis=0)
        for k in range(len(uc)):
            for l in range(len(val_i)):
                if(uc[k] == l):
                    edge_map[point_i][val_i[l]].append(val_j[k])

    else:
        hc = all_cluster_table.argmin(axis=1)
        for k in range(len(val_i)):
            edge_map[point_i][val_i[k]].append(val_j[hc[k]])
    
    for idx,val in enumerate(val_i):
            if(edge_map[point_i][val] == []):
                qc = all_cluster_table[idx].argmin()
                print(qc)
                edge_map[point_i][val].append(val_j[qc])
                # print(all_cluster_table[idx].argmin())

### Use This Block for the minimum path


keys = list(date_country_map.keys())
n_edge_map = {}
next_ = 0
for idx in range(len(date_country_map)-1):
    point_i = keys[idx]
    point_j = keys[idx+1]
    
    file_list_i = os.listdir(all_cluster_direc+"/"+formatter(point_i))
    file_list_j = os.listdir(all_cluster_direc+"/"+formatter(point_j))

    val_i = file_array_formatter(file_list_i)
    val_j = file_array_formatter(file_list_j)
    n_edge_map[point_i] = {}
    if(len(val_i) == 0 or len(val_j) == 0):
        continue

    print("doing->",len(val_i),len(val_j))
    all_cluster_table = np.full((len(file_list_i),len(file_list_j)),-90)
    # print(all_cluster_table.shape)
    count = 0
    for i,file1 in enumerate(file_list_i):
        region = file_name_checker(file1)
        bleh1 = pd.read_csv(all_cluster_direc+"/"+formatter(point_i)+"/"+file1)
        for j,file2 in enumerate(file_list_j):
            # # if(all_cluster_table)
            # if ((region not in file2) and (all_cluster_table[i][j] == -90)):
            count+=1
            bleh2 = pd.read_csv(all_cluster_direc+"/"+formatter(point_j)+"/" +file2)
            # print(file1,file2)
            all_cluster_table[i][j] = (get_cluster_dist(bleh1,bleh2)/(bleh1.shape[0]*bleh2.shape[0]))

    # # print(point_i,point_j,all_cluster_table.shape)
  
    for kl in val_i:
        n_edge_map[point_i][kl] = []
    uc =  np.unravel_index(np.argmin(all_cluster_table, axis=None), all_cluster_table.shape) # np.where(all_cluster_table == np.min(all_cluster_table)) #all_cluster_table.argmin()
    print(uc)
    if(next_ != uc[0]):
        print("such case")
        # print(next_,uc[0])
        uc = list(np.unravel_index(np.argmin(all_cluster_table[next_]),all_cluster_table.shape))
        uc[0] = next_
        print(uc)
    
    next_ = uc[1]
    n_edge_map[point_i][val_i[uc[0]]].append(val_j[uc[1]])


pprint(n_edge_map)

pprint(final_input_map)

import datetime

############### generating training set #################################
INF = 99999999999
total_time_series = 5
time_series_all = {}
total_unique_seq = 0

BD_first_date = datetime.date(2020, 3, 9)
IN_first_date = datetime.date(2020, 1, 31)
PK_first_date = datetime.date(2020, 1, 30)
MN_first_date = datetime.date(2020, 3, 24)
# count = 0
for i in range(INF):
    one_time_series = ""
    current_country = "China"
    date_count = 0
    for date in final_input_map.keys():
        if date == BD_first_date:
          break
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
        if date in n_edge_map.keys():        
            total_next_country = len(n_edge_map[date][current_country])
            next_country_idx = randrange(total_next_country)
            current_country = n_edge_map[date][current_country][next_country_idx]

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

with open(OUTPUT_FOLDER+'test_time_series_BD.txt', 'w') as f:
    for item in list_time_series_all:
        f.write("%s\n" % item)

len(list(time_series_all)[0])