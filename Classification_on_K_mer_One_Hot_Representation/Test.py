#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from itertools import permutations,combinations_with_replacement


import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
# from tf.keras import load_model

from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# In[ ]:


df_test=pd.read_csv('Test_gisaid_LS.csv')
df_train=pd.read_csv('Train_gisaid_LS.csv')


# In[ ]:


train_sequences=list(df_train["Sequence"])
print(len(train_sequences))
y_indicator=list(df_train["Indicator"])


# In[ ]:


K_MER_LENGTH = 3
NUMBER_OF_K_MERS=np.power(4,K_MER_LENGTH)


# In[ ]:


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
        
        # all_sequences_k_mer.append(seq_k_mer_list)
    return seq_k_mer_list



def get_motifs_V2(length):
    lst = ['A','C','G','T','N']
    # els = [list(x) for x in itertools.combinations(lst, length)]
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


# In[ ]:


length_max=0
for i in range(0,len(train_sequences)):
  k_mer_list=get_motifs(K_MER_LENGTH,train_sequences[i])
  if len(k_mer_list)>length_max:
    length_max=len(k_mer_list)
print(length_max)
MAX_LEN=length_max



# In[ ]:


all_combination_list=get_all_combs('ATCG',K_MER_LENGTH)
all_comb_dict={}
for i in range(0,len(all_combination_list)):
  all_comb_dict[all_combination_list[i]]=i
print(all_comb_dict)


# In[ ]:


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

    yield np.array(sequences_2D_Mats)
    
      


# In[ ]:


BS = 1


LOSS_FN = 'binary_crossentropy'  
LEARNING_RATE = 0.000001   # 0.0001, 0.00005, 0.00003


# In[ ]:


model=tf.keras.models.load_model("best_model_3_AlexNet_BS_10_E_135.h5")
#model=tf.keras.models.load_model("best_model_4_AlexNet_BS_10_E_100.h5")
#model=tf.keras.models.load_model("best_model_5_AlexNet_BS_10_E_100.h5")


# In[ ]:


test_sequences=list(df_test["Sequence"])
y_indicator_test=list(df_test["Indicator"])


# In[ ]:


new_test_sequence_list=[]
new_y_ind_list=[]
for i in range(0,len(test_sequences)):
    k_mer_list=get_motifs(K_MER_LENGTH,test_sequences[i])
    if len(k_mer_list)<=MAX_LEN:
        new_test_sequence_list.append(test_sequences[i])
        new_y_ind_list.append(y_indicator_test[i])
    else:
        print(i)
        print(len(k_mer_list))


# In[ ]:


X_test=np.array(new_test_sequence_list)
y_test=np.array(new_y_ind_list)
num_test=len(new_test_sequence_list)
print(num_test)


# In[ ]:


testGen = sequence_generator(X_test,MAX_LEN, all_comb_dict,BS,y_test)
y_pred = model.predict_generator(testGen,steps=num_test//BS)


# In[ ]:


y_test=y_test[:y_pred.shape[0]]
y_test=y_test.reshape(y_test.shape[0],1)
print(y_test.shape)


# In[ ]:


y_pred_before_rounding=y_pred
y_pred=np.round(y_pred)
print(y_pred.shape)


# In[ ]:


match=0
fn=0
fp=0
tp=0
tn=0
for i in range(y_pred.shape[0]):    
  if(y_pred[i]==y_test[i]):
    match+=1
    if (y_test[i]==1.0 and y_pred[i]==1.0):
      tp+=1
    elif (y_test[i]==0.0 and y_pred[i]==0.0):
      tn+=1
  elif (y_test[i]==0.0 and y_pred[i]==1.0):
    fp+=1
  elif (y_test[i]==1.0 and y_pred[i]==0.0):
    fn+=1

print("fp,fn,tp,tn:"+str(fp)+","+str(fn)+","+str(tp)+","+str(tn))
print(match)
acc=match*100.0/y_pred.shape[0]
precision=tp*1.0/(tp+fp)
recall=tp*1.0/(tp+fn)
f1_score=(2*recall*precision)/(recall+precision)
print("Accuracy: "+str(acc))
print("precision: "+str(precision))
print("recall: "+str(recall))
print("f1_score: "+str(f1_score))


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


fpr, tpr, thresholds_keras = roc_curve(y_test, y_pred_before_rounding)
_auc_ = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='ROC-One-Hot-k-'+str(K_MER_LENGTH)+' (area = {:.3f})'.format(_auc_))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('ROC_One_hot_k_'+str(K_MER_LENGTH)+'.png')
plt.show()



