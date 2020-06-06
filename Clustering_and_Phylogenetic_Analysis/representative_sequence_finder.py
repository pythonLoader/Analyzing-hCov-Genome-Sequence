import sklearn.cluster as cluster
import pandas as pd
import numpy as np 
# from Bio import SeqIO,pairwise2
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
# import matplotlib.pyplot as plt
import os,sys
from sklearn.cluster import KMeans
import sklearn
import string
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.metrics import pairwise_distances_argmin_min


method = ""
dataset = ""
direc = ""

def clustering(ACC_ids_list,score_mat,num_seqs,num_clusters):

   
    ids_list = ACC_ids_list
   
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = score_mat.index.values
   
    scores = score_mat.to_numpy()
    # print(scores.shape)

    kmeans = cluster.KMeans(num_clusters)
    results = kmeans.fit(scores)

    cluster_map['cluster'] = results.labels_
    labels = results.labels_
    

    # print(num_clusters)
    centers = results.cluster_centers_

    closest, _ = pairwise_distances_argmin_min(centers, scores)
    # print(closest)
    # print(ids_list[closest[0]])
    clusters = [[] for i in range(num_clusters)]

    info_handle = open(direc+"/"+dataset+"_center.txt","w")
    info_handle.write(ids_list[closest[0]])
    info_handle.close()

    for i in range(0,num_seqs):
        clusters[labels[i]].append(ids_list[i])


def MDS_on_distance_matrix(matrix):
    # score_matrix = StandardScaler().fit_transform(matrix)
    score_matrix = matrix
    model = MDS(n_components=2,dissimilarity="precomputed",random_state=1)
    matrix_transformed = model.fit_transform(score_matrix)
    # print(matrix_transformed.shape)
    

def PCA_on_distance_matrix(scores,num_clusters):
    matrix = scores
    # matrix = scores.to_numpy()
    print("Starting PCA")
    PCA_model = PCA(n_components=2)
    # PCA_model.fit(matrix)
    scaled_matrix = StandardScaler().fit_transform(matrix)
    PCA_model.fit(scaled_matrix)
    # print(PCA_model.components_)
    # print(PCA_model.explained_variance_)
    # print(PCA_model.explained_variance_ratio_)

    matrix_transformed = PCA_model.transform(scaled_matrix)
    # print(matrix_transformed.shape)
    # num_clusters = silhoutte_method(matrix_transformed)
    df = pd.DataFrame(matrix_transformed,columns=['PC_1','PC_2'])
    return df
    # print(num_clusters)
    


def main():
  
    rem = os.listdir('All_Countries_Distance_Matrix')
    main_direc = 'All_Countries_Distance_Matrix'
    global direc
    direc = "All_Countries_Representative_Seq"
    print("working with -> "+direc)
    if not os.path.exists(direc):
        os.mkdir(direc)

    for cont_ in rem:
        global dataset,method
        
        dataset = cont_.split("_")[0] #Change this
        method = "Represtative_seq"

        c_time = time()

        csv_file = main_direc +"/"+dataset+"_distance_matrix.csv"

        matrix = pd.read_csv(csv_file)
        print(matrix.shape)
        print("Time for loading dataset ->",time()-c_time)
        ################
        if(matrix.shape[0] == 1):
            info_handle = open(direc+"/"+dataset+"_center.txt","w")
            info_handle.write(matrix.columns[0])
            info_handle.close()
            continue
        num_clusters = 1
        
        df = PCA_on_distance_matrix(matrix,num_clusters)
        num_seqs = matrix.shape[0]
        clustering(matrix.copy().columns,df,num_seqs,num_clusters)
        

if __name__ == '__main__':
    main()