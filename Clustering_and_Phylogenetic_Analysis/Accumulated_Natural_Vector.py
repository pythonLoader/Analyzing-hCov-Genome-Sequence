import itertools
import os
import numpy as np
import sys
import pandas as pd
from collections import Counter
from itertools import combinations

def find_covariance(vect1,vect2):
    cov_np = np.cov(vect1,vect2)
    mean_1 = np.mean(vect1)
    mean_2 = np.mean(vect2)
    n1 = np.max(vect1)
    n2 = np.max(vect2)
    assert vect1.shape[0] == vect2.shape[0]
    sum = 0
    for val1,val2 in zip(vect1,vect2):
        sum += (val1-mean_1)*(val2-mean_2)
    sum = sum/(n1*n2)
    covariance = sum
    return covariance

def accumulated_nv_generator(seq):
    
    Accumulated_Natural_Vector = np.zeros(18)

    U = {}
    U["A"],U["T"],U["C"],U["G"] = np.zeros((4,len(seq)))
    n_alpha = Counter(seq)
    count = 0
    for key in n_alpha.keys():
        Accumulated_Natural_Vector[count] = n_alpha[key]
        count+=1 
    # print(n_alpha["A"])
    for idx,c in enumerate(seq):
        if c == "A":
            U["A"][idx] = 1
        elif c == "T":
            U["T"][idx] = 1
        elif c == "G":
            U["G"][idx] = 1
        elif(c == "C"):
            U["C"][idx] = 1
        else:
            print("Vodox->",c,idx)
    U_accumulated = {}
    for key in U.keys():
        U_accumulated[key] = np.cumsum(U[key])
    
    # print(U_accumulated["A"])
    # print(U_accumulated["C"])
    zeta = {}
    for key in U_accumulated.keys():
        zeta[key] = np.sum(U_accumulated[key])/n_alpha[key]
        Accumulated_Natural_Vector[count] = zeta[key]
        count+=1

    Divergence = {}
    for key in U_accumulated.keys():
        sum = 0
        mean_alpha = np.mean(U_accumulated[key])
        for val in U_accumulated[key]:
            sum+= (((val - mean_alpha)/(n_alpha[key]))**2)
        Divergence[key] = sum 
        Accumulated_Natural_Vector[count] = Divergence[key]
        count+=1
        # sum = sum/((n_alpha[key])**2)
        # print(key,sum,np.var(U_accumulated[key]))
    key_combo = list(combinations(U_accumulated.keys(),2))
    sorted_key_combo = sorted(key_combo,key=lambda x: (x[0], x[1]))

    for tup in sorted_key_combo:
        cov = find_covariance(U_accumulated[tup[0]],U_accumulated[tup[1]])
        Accumulated_Natural_Vector[count] = cov
        count+=1
    assert count == 18
    # print(Accumulated_Natural_Vector)
    return Accumulated_Natural_Vector

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
    files = os.listdir(main_dir)
    if not os.path.exists("All_Countries_ACC_Vects"):
        os.mkdir("All_Countries_ACC_Vects")
    if not os.path.exists("All_Countries_Distance_Matrix"):
        os.mkdir("All_Countries_Distance_Matrix")
    
    for file_ in files:
        inp_file = main_dir + "/"+file_
        c_name = file_.split(".")[0]
        df = pd.read_csv(inp_file)
        
        sequences = df["Sequence"]
        print("Started working with ->",file_,"n_seq->",sequences.shape)

        acc_vects = []
        count = 0
        for seq in sequences:
        
            acc_vects.append(accumulated_nv_generator(seq))
            if(count%100 == 0):
                print(count)
            count +=1
            # print(count)

        acc_vects = np.array(acc_vects)
        print(acc_vects.shape)
        cont_ = np.array(acc_vects)
        # print(cont_.shape)
        ID_arr = df["Accession ID"]
        # print(ID_arr.shape)

        ID_col = pd.Series(ID_arr)
        Vector_col = pd.Series(cont_.tolist())

        frame = {'Accession ID': ID_col,'Vector':Vector_col}
        df_final = pd.DataFrame(frame)
        direc_1 = "All_Countries_ACC_Vects"
        df_final.to_csv(direc_1+"/"+c_name+"_Accumulated_Fast_Vector.csv",index=False)
        
        direc_2 = "All_Countries_Distance_Matrix"
        matrix = euclidean(cont_,len(cont_))
        print(matrix.shape)
        final_df = pd.DataFrame(matrix,columns=ID_arr)
        final_df.to_csv(direc_2+"/"+c_name+"_accumulated_distance_matrix.csv",index=False)

def main():
    create_Vects_and_Country_wise_distance_mat()

if __name__ == "__main__":
    main()