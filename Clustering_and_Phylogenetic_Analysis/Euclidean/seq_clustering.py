import sklearn.cluster as cluster
import pandas as pd
import numpy as np 
# from Bio import SeqIO,pairwise2
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib.pyplot as plt
import os,sys
from sklearn.cluster import KMeans
import sklearn
import string
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.metrics import pairwise_distances_argmin_min

# import networkx as nx

method = ""
dataset = ""
direc = ""

def silhoutte_method(obs):
    silhouette_score_values=list()
 
    NumberOfClusters=range(2,11)
     
    for i in NumberOfClusters:
        
        classifier=KMeans(i,init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True)
        classifier.fit(obs)
        labels= classifier.predict(obs)
        print("Number Of Clusters:",i)
        
        print("Silhouette score value",sklearn.metrics.silhouette_score(obs,labels ,metric='euclidean', sample_size=None, random_state=None))
        silhouette_score_values.append(sklearn.metrics.silhouette_score(obs,labels ,metric='euclidean', sample_size=None, random_state=None))
    
    option = "silhouette"
    plot_name = direc +"/"+ dataset + "_" + method +"_" + option + ".png"
     
    plt.plot(NumberOfClusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters ")
    plt.savefig(plot_name,transparent=False)
    plt.close()
    # plt.show()
     
    Optimal_NumberOf_Components=NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print("Optimal number of components is:",Optimal_NumberOf_Components)
    return Optimal_NumberOf_Components

def elbow_method(X):

    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    K = range(1,10) 
      
    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k).fit(X) 
        kmeanModel.fit(X)     
          
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'),axis=1)) / X.shape[0]) 
        inertias.append(kmeanModel.inertia_) 
      
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'),axis=1)) / X.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 


    for key,val in mapping1.items(): 
        print(str(key)+' : '+str(val)) 


    plt.plot(K, inertias, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Inertia') 
    plt.title('The Elbow Method using Inertia') 
    # 
    option = "elbow"

    plot_name = direc +"/"+ dataset + "_" + method +"_" + option + ".png"
    plt.savefig(plot_name,transparent=False) 
    plt.close()
    # plt.show()


def visualize(scores):
    # print(os.getenv("DISPLAY"))
   
    
    # # plt.imshow(score_mat)
    # # plt.show()
    # dt = [('len', float)]
    # A = score_mat
    # A = A.view(dt)

    # G = nx.from_numpy_matrix(A)
    # G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))    

    # G = nx.drawing.nx_agraph.to_agraph(G)

    # G.node_attr.update(color="red", style="filled")
    # G.edge_attr.update(color="blue", width="2.0")

    # G.draw('out.png', format='png', prog='neato')

    score_mat = scores.to_numpy()
    print(score_mat.shape)
    elbow_method(score_mat)
    num_clusters = silhoutte_method(score_mat)
    return num_clusters

    # x = score_mat[:,0]
    # y = score_mat[:,1]
    # plt.scatter(x,y)
    # option = "visualize"
  


    # plot_name = direc +"/"+ dataset + "_" + method +"_" + option + ".png"
    # plt.savefig(plot_name,transparent=False)
    # plt.close()
    
def plot_final_cluster(x,y,centers,labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x,y,c=labels)
    for i,j in centers:
        ax.scatter(i,j,c='red',marker='+')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter)
    option = "final_cluster"
    plot_name = direc +"/"+ dataset + "_" + method +"_" + option + ".png"
    fig.savefig(plot_name,transparent=False)
    plt.close()
    # fig.close()

    # plt.show()
    

def clustering(ACC_ids_list,score_mat,num_seqs,num_clusters):

    # handle = open(seq_file, "r+")

    # records = list(SeqIO.parse(handle, "fasta"))
    # num_seqs = len(records)
    ids_list = ACC_ids_list
    # ids_list=score_mat.columns
    # print(ids_list)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = score_mat.index.values
   
    scores = score_mat.to_numpy()
    print(scores.shape)

    kmeans = cluster.KMeans(num_clusters)
    results = kmeans.fit(scores)

    cluster_map['cluster'] = results.labels_
    labels = results.labels_
    

    print(num_clusters)
    centers = results.cluster_centers_

    closest, _ = pairwise_distances_argmin_min(centers, scores)
    print(closest)
    print(ids_list[closest[0]])
    # print(labels)
    clusters = [[] for i in range(num_clusters)]

    info_handle = open(direc+"/"+dataset+"_center.txt","w")
    info_handle.write(ids_list[closest[0]])
    info_handle.close()

    for i in range(0,num_seqs):
        clusters[labels[i]].append(ids_list[i])


    ################################################
    #Scatter plotting the cluster
    x = scores[:,0]
    y = scores[:,1]

    plot_final_cluster(x,y,centers,labels)
  

    ################################################
    # for i in range(0,len(clusters)):
    #     info_handle = open(direc+"/cluster_."+str(i)+".txt","w")
    #     info_handle.write("Centre\n")
    #     info_handle.write(ids_list[closest[0]])
        
        # for ids in clusters[i]:
        #     # info_handle.write("\n")
        #     info_handle.write(ids)
        #     info_handle.write("\n")
    



    # for i in range(0,len(clusters)):
    #     output_handle = open(direc+"/c."+str(i)+".fasta", "w")
    #     info_handle = open(direc+"/c_info."+str(i)+".txt","w")
    #     print(len(clusters[i]))
        
    #     SeqIO.write(clusters[i], output_handle, "fasta")
    #     # print(clusters[i])
    #     for cluster_info in clusters[i]:
    #         # print("i-> ", i, " length->", len(cluster_info.seq))
    #         info_handle.write("\n")
    #         info_handle.write(str(cluster_info))
    #         info_handle.write("\n")
    #     output_handle.close()
    #     info_handle.close()





def MDS_on_distance_matrix(matrix):
    # score_matrix = StandardScaler().fit_transform(matrix)
    score_matrix = matrix
    model = MDS(n_components=2,dissimilarity="precomputed",random_state=1)
    matrix_transformed= model.fit_transform(score_matrix)
     

    plt.scatter(matrix_transformed[:,0], matrix_transformed[:,1],alpha=0.1,color='black')
    plt.show()

def PCA_on_distance_matrix(scores,num_clusters):
    matrix = scores
    # matrix = scores.to_numpy()
    print("Starting PCA")
    PCA_model = PCA(n_components=2)
    num_seqs = matrix.shape[0]
    # PCA_model.fit(matrix)
    scaled_matrix = StandardScaler().fit_transform(matrix)
    PCA_model.fit(scaled_matrix)


    # print(PCA_model.components_)
    # print(PCA_model.explained_variance_)
    # print(PCA_model.explained_variance_ratio_)

    matrix_transformed = PCA_model.transform(scaled_matrix)
    print(matrix_transformed.shape)
    # num_clusters = silhoutte_method(matrix_transformed)
    df = pd.DataFrame(matrix_transformed,columns=['PC_1','PC_2'])
    # print(num_clusters)
    clustering(matrix.copy().columns,df,num_seqs,num_clusters)
   


    # plt.scatter(matrix_transformed[:,0], matrix_transformed[:,1],alpha=0.1, color='black')
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')

    # plt.show()

def main():
    # if(len(sys.argv) < 2):
    #     print("Format -> <handle> <method_name>")
    #     exit(1)

    # seq_file = "sequences_new_complete.fasta" #Change this
    # rem = ['Asia','Oceania','Europe','North_America','Others']
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

        
        
        ################## Change this also
        # csv_file = "GenBank_New/genu_"+method+"_o.csv"
        # if(method == "kl" or method == "lcc"):
        #     csv_file = "GenBank_New/genu_"+method+"_f.csv"
        c_time = time()

    #     n_seq = encode()
        csv_file = main_direc +"/"+dataset+"_accumulated_distance_matrix.csv"

        matrix = pd.read_csv(csv_file)
        print(matrix.shape)
        print("Time for loading dataset ->",time()-c_time)
        ################
        if(matrix.shape[0] == 1):
            info_handle = open(direc+"/"+dataset+"_center.txt","w")
            info_handle.write(matrix.columns[0])
            info_handle.close()
            continue

        
        
        # num_clusters = visualize(matrix)
        num_clusters = 1
        # visualize(matrix)
        # num_clusters = 2
        print("Done with visualizing")

        PCA_on_distance_matrix(matrix,num_clusters)
        # MDS_on_distance_matrix(matrix.to_numpy())

        # # num_clusters = 4
        # clustering(matrix,seq_file,num_clusters)
        # print("Done with clustering")


if __name__ == '__main__':
    main()