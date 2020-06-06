import itertools
import os
import numpy as np
import sys
import pandas as pd
from collections import Counter

def encode(seq,change_map):
    n_seq = []
    for base in seq:
        if base in change_map:
            n_seq.append(change_map[base])
        else:
            print("More than 4 symbols! Problem!!!")
            print(base)
            print(len(seq),len(n_seq))
    
    assert len(seq) == len(n_seq)
    return n_seq

def R_Y_coding(seq):
    change_map = {'A':'R','G':'R','C':'Y','T':'Y'}
    
    
    n_seq = encode(seq,change_map)
    unique,count = np.unique(n_seq,return_counts=True)
    n_r = count[0]
    n_y = count[1]
    # print(unique,count)
    ry_encode_seq = ''.join(n_seq)
    # print(ry_encode_seq)
    return (ry_encode_seq,n_r,n_y)

def M_K_coding(seq):
    change_map = {'A':'M','G':'K','C':'M','T':'K'}

    n_seq = encode(seq,change_map)
    counts = Counter(n_seq)
    # print(counts)
    n_m = counts['M']
    n_k = counts['K']
    mk_encode_seq = ''.join(n_seq)
    return (mk_encode_seq,n_m,n_k)

def S_W_coding(seq):
    change_map = {'A':'W','G':'S','C':'S','T':'W'}

    n_seq = encode(seq,change_map)
    counts = Counter(n_seq)
    # print(counts)
    n_s = counts['S']
    n_w = counts['W']
    sw_encode_seq = ''.join(n_seq)
    return (sw_encode_seq,n_s,n_w)

def get_mean_position(seq,n_1,c):
    length = len(seq)
    sum = 0
    for idx in range(length):
        if(seq[idx] == c):
            sum+= (idx*(1.0/n_1))
    return sum

def get_variance(seq,n_1,meu,c):
    length = len(seq)
    sum = 0
    for i in range(length):
        if(seq[i] == c):
            sum += (((i-meu)**2)*1.0)/(n_1*length)
    return sum

def get_NFV(seq):
    (ry_encode_seq,n_r,n_y)  = R_Y_coding(seq)
    (mk_encode_seq,n_m,n_k) = M_K_coding(seq)
    (sw_encode_seq,n_s,n_w) = S_W_coding(seq)

    meu_r = get_mean_position(ry_encode_seq,n_r,'R')
    meu_y = get_mean_position(ry_encode_seq,n_y,'Y')

    meu_m = get_mean_position(mk_encode_seq,n_m,'M')
    meu_k = get_mean_position(mk_encode_seq,n_k,'K')

    meu_s = get_mean_position(sw_encode_seq,n_s,'S')
    meu_w = get_mean_position(sw_encode_seq,n_w,'W')

    D_r = get_variance(ry_encode_seq,n_r,meu_r,'R')
    D_y = get_variance(ry_encode_seq,n_y,meu_y,'Y')

    D_m = get_variance(mk_encode_seq,n_m,meu_m,'M')
    D_k = get_variance(mk_encode_seq,n_k,meu_k,'K')

    D_s = get_variance(sw_encode_seq,n_s,meu_s,'S')
    D_w = get_variance(sw_encode_seq,n_w,meu_w,'W')
    
    Fast_vector = [n_r,meu_r,D_r,n_y,meu_y,D_y,n_m,meu_m,D_m,n_k,meu_k,D_k,n_s,meu_s,D_s,n_w,meu_w,D_w]
    assert len(Fast_vector) == 18
    return Fast_vector

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
    if not os.path.exists("All_Countries_NFV_Vects"):
        os.mkdir("All_Countries_NFV_Vects")
    if not os.path.exists("All_Countries_Distance_Matrix"):
        os.mkdir("All_Countries_Distance_Matrix")

    for file_ in files:
        inp_file = main_dir + "/"+file_
        c_name = file_.split(".")[0]
        df = pd.read_csv(inp_file)
        
        
        print("Started working with ->",file_)
        sequences = df["Sequence"]
        final_list = []
        count = 0
        for seq in sequences:
            Fast_vector = get_NFV(seq)
            final_list.append(Fast_vector)
            if(count%100 == 0):
                print(count)
            count +=1

        acc_vects = np.array(final_list)
        print(acc_vects.shape)
        cont_ = np.array(acc_vects)
        
        ID_arr = df["Accession ID"]
        ID_col = pd.Series(ID_arr)
        Vector_col = pd.Series(cont_.tolist())
        frame = {'Accession ID': ID_col,'Vector':Vector_col}
        df_final = pd.DataFrame(frame)
        direc_1 = "All_Countries_NFV_Vects"
        df_final.to_csv(direc_1+"/"+c_name+"_Novel_Fast_Vector.csv",index=False)
        direc_2 = "All_Countries_Distance_Matrix"
        matrix = euclidean(cont_,len(cont_))
        print(matrix.shape)
        final_df = pd.DataFrame(matrix,columns=ID_arr)
        final_df.to_csv(direc_2+"/"+c_name+"_distance_matrix.csv",index=False)


def main():
    create_Vects_and_Country_wise_distance_mat()

if __name__ == "__main__":
    main()