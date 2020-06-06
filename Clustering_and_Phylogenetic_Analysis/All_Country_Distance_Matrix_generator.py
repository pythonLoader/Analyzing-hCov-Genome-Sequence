import pandas as pd 
import numpy as np 
import sys,os
import argparse

direc = 'All_Countries_Representative_Seq'

def Representative_Sequence_Info_extractor(df,method,ind_):
    direc = 'All_Countries_Representative_Seq'
    # Other_Info_File = "All_Train_Test_Gisaid.csv"

    VECTS_direc = 'All_Countries_'+ind_+'_Vects'
    Other_Info_df = df.copy()
    # Other_Info_df = pd.read_csv(Other_Info_File)

    df_list = []
    # cols = ['Country','Accession ID', 'Fast Vector', 'Virus name', 'Location',
    #        'Collection date', 'Death', 'Indicator', 'Sequence']
    cols = ['Country','Accession ID', 'Vector', 'Virus name', 'Location',
        'Collection date', 'Sequence']
    for file_ in os.listdir(direc):
        typ = file_.split(".")[1]
        if(typ == 'txt'):
            cont_ = file_.split("_")[0]
            print("working with->",cont_)
            Vect_file = VECTS_direc +"/"+cont_+"_"+method +".csv"
            c_fl = open(direc+"/"+file_,"r")
            center_id = c_fl.read()
            print("center",center_id)
            Fast_vector_df = pd.read_csv(Vect_file)

            final_df_1 = Fast_vector_df.loc[Fast_vector_df["Accession ID"] == center_id]
            final_df_2 = Other_Info_df.loc[Other_Info_df["Accession ID"] == center_id]

            if(cont_ == "United Kingdom" or cont_ == "Korea"):
                continue
            # frame = {'Accession ID':pd.Series(center_id),'Country':pd.Series(cont_)}
            # final_df_3 = pd.DataFrame(frame)
            # print(final_df_1.shape)
            # print(final_df_2.shape)
            final_df = pd.merge(final_df_1, final_df_2, on='Accession ID')
            final_df["Country"] = cont_
            final_df = final_df[cols]
            df_list.append(final_df)

    f_df = pd.concat(df_list,ignore_index=True)
    print(f_df.shape)
    return f_df


def all_country_distance_table(f_df):
    vect = f_df['Vector']
    cont_ = f_df['Country']
    final_vect_len = f_df.shape[0]
    print(final_vect_len)
    final_vect = np.full((final_vect_len,final_vect_len),-89)
    for i in range(final_vect_len):
        for j in range(final_vect_len):
            # print("Doing -> ",cont_[i],cont_[j])
            var_1 = np.asarray(vect[i].replace("[","").replace("]","").replace(' ','').split(",")).astype(np.float)
            var_2 = np.asarray(vect[j].replace("[","").replace("]","").replace(' ','').split(",")).astype(np.float)
            final_vect[i][j] = np.linalg.norm((var_1-var_2),ord=2)
            # print(final_vect[i][j])

    # print(final_vect)
    return (np.array(cont_),final_vect)



def Distance_Matrix_Creation(f_df,method):
    # f_df = pd.read_csv("All_country_representative_seq_info.csv")
    (cont_,final_vect) = all_country_distance_table(f_df)
    print(cont_)
    final_col = []
    final_col.append("")
    final_col.extend(cont_)
    cont_ = cont_.reshape(cont_.shape[0],1)
    target = np.hstack((cont_, final_vect))
    print(target.shape)
    ekdom_final_df = pd.DataFrame(target,columns=final_col)
    # print(ekdom_final_df.shape)
    ekdom_final_df.to_csv("Final_distance_matrix_"+method+".csv",index=False)

def main():
    methods = ('Euclidean','cosine','Novel_Fast_Vector','Accumulated_Fast_Vector')
    parser = argparse.ArgumentParser(description="Final Distance Matrix Generator")
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--label','-l',help='Label on which parameter (Death/CFR_confirmed_cases/CFR_Recovery/CFR_Infrastructure)',required=True,type=str)
    required.add_argument('--method','-m',type=str, required=True, choices = methods,help='Available methods: ' +"'"+"', '".join(methods) +"'")
    
    
    pars = parser.parse_args()
    label = pars.label
    method = pars.method
    if method == "Novel_Fast_Vector":
        ind_ = "NFV"
    elif method == "Accumulated_Fast_Vector":
        ind_ = "ACC"
    else:
        ind_ = " Euclidean"
    # print(label)
    required_file = "../Input/All_labelled_by_" + label +".csv"
    df = pd.read_csv(required_file)
    print(df.shape)
    f_df = Representative_Sequence_Info_extractor(df,method,ind_)
    Distance_Matrix_Creation(f_df,method)

if __name__ == "__main__":
    main()