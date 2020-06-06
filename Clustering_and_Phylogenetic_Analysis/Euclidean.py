import itertools
import os
import numpy as np
import sys
from pprint import pprint
import pandas as pd
import random
from collections import Counter

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
    for seq in sequences_list:
        for i in range(0, len(seq)-length+1):
            word = seq[i:i+length]
            if word not in d:
                d[word] = index
                index += 1 
    return d

def get_headers(input_file):
    list_of_ids=[]
    for line in open(input_file):
        if line.startswith('>'):
            line = line.replace('>','').split()
            list_of_ids.append(line[0])
    return list_of_ids

def calculate_occurrences(length, input_file):
    df = pd.read_csv(input_file)
    sequences_list = df["Sequence"]  #get_sequences(input_file)
    d = get_motifs(length,sequences_list)
    print("Done with motif finding")   
    rows_num = len(sequences_list)
    cols_num = len(d)
    data = np.zeros(shape=(rows_num,cols_num))
    for row_idx, seq in enumerate(sequences_list):
        for i in range(0, len(seq)-length+1):
            word = seq[i:i+length]
            col_idx = d[word]
            data[row_idx, col_idx] += 1
    return data

def calculate_frequencies(occurrences_list,seqs_number):
    frequencies_list =[]
    for i in range(0,seqs_number):
        frequencies_list.append(occurrences_list[i,:]/np.sum(occurrences_list[i,:]))
    return np.vstack(frequencies_list)

def minkowski(list_,seqs_number,exponent): 
    matrix = np.zeros([seqs_number, seqs_number])
    for i, j in itertools.combinations(range(0,seqs_number),2):
         matrix[i][j]= matrix [j][i] = np.linalg.norm((list_[i,:]-list_[j,:]),ord=exponent)
        #  (np.sum((np.absolute(list_[i,:] - list_[j,:]))**exponent))**(1.0/float(exponent))
    return matrix
def euclidean(list_,seqs_number):
    return minkowski(list_,seqs_number,2)

def create_Vects_and_Country_wise_distance_mat():
    main_dir = 'All_Countries_Splitted'
    files = os.listdir('All_Countries_Splitted')
    direc_1 = "All_Countries_Euclidean_Vects"
    if not os.path.exists(direc_1):
        os.mkdir(direc_1)
    if not os.path.exists("All_Countries_Distance_Matrix"):
        os.mkdir("All_Countries_Distance_Matrix")


    for file_ in files:
        
        inp_file = main_dir + "/"+file_
        c_name = file_.split(".")[0]
        # if(c_name in contries_):
        #     continue
        df = pd.read_csv(inp_file)
        
        
        print("Started working with ->",file_)
        sequences = df["Sequence"]
        print("n_seq->",sequences.shape)
        final_list = []
        count = 0
        # for seq in sequences:
        final_list = calculate_occurrences(length=3,input_file=inp_file)
        #     n_seq = encode()
        # for seq in sequences:
        #     Fast_vector = get_NFV(seq)
        #     final_list.append(Fast_vector)
        #     if(count%100 == 0):
        #         print(count)
        #     count +=1
            # print(Fast_vector)
        
        acc_vects = cont_ = np.array(final_list)
        print(acc_vects.shape)
        # cont_ = np.array(acc_vects)
        # print(cont_.shape)
        ID_arr = df["Accession ID"]
        # print(ID_arr.shape)
        ID_col = pd.Series(ID_arr)
        Vector_col = pd.Series(cont_.tolist())
        frame = {'Accession ID': ID_col,'Vector':Vector_col}
        df_final = pd.DataFrame(frame)
        # direc_1 = "All_Countries_Euclidean_Vects"
        df_final.to_csv(direc_1+"/"+c_name+"_Euclidean.csv",index=False)
        direc_2 = "All_Countries_Distance_Matrix"
        matrix = euclidean(cont_,len(cont_))
        print(matrix.shape)
        final_df = pd.DataFrame(matrix,columns=ID_arr)
        final_df.to_csv(direc_2+"/"+c_name+"_distance_matrix.csv",index=False)


def main():
    create_Vects_and_Country_wise_distance_mat()

if __name__ == "__main__":
    main()