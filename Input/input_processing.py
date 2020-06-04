import pandas as pd
import numpy as np
from Bio import SeqIO
import argparse
from sklearn.model_selection import train_test_split
import random
old_flag = False

def read_fasta(input_fasta):
    # Read Sequences from the fasta files
    # Returns an (Accession Id, Sequence) Dictionary
    count=0
    id_seq_map = {}
    for seq_record in SeqIO.parse(input_fasta,"fasta"):
        id_ = seq_record.description.split("|")[1]
        id_seq_map[id_] = str(seq_record.seq)
        count+=1
    print(count)
    return id_seq_map

def GisAID_Processing(df_name,original_df,Label):
    # Use the Preprocessed Dataset Labelling Helper to Label the supplied dataset
    # Creates the csv file of the dataframe supplied with labels(Indicator)
    # Returns the dataframe for later use in some use-cases.
    label_helper_df = pd.read_csv("Dataset_Labelling_Helper.csv")
    print(original_df.shape)
    original_df = original_df.dropna()

    locations = original_df["Location"]
    countries = label_helper_df["Countries"]
    deaths = label_helper_df["Death"]
    ind_ = "Indicator_" + Label

    indicator = label_helper_df[ind_]

    death_arr = np.full(locations.shape[0],-1)
    indicator_arr = np.full(locations.shape[0],-1)

    count = 0

    for l_idx,l_value in enumerate(locations):
        l_value = str(l_value).split("/")
        for c_idx,c_value in enumerate(countries):

            if(c_value in str(l_value).encode('ascii', 'ignore').decode('ascii')): #This is required for handling ASCII conversion problem
                death_arr[l_idx] = deaths[c_idx]
                indicator_arr[l_idx] = indicator[c_idx]
                count+=1

    original_df["Death"] = death_arr
    original_df["Indicator"] = indicator_arr

    unique, counts = np.unique(indicator_arr, return_counts=True)
    print(unique,counts)
    permitted = [0,1]

    good_df = original_df.loc[original_df["Indicator"].isin(permitted)]

    if(original_df.shape != good_df.shape):

        rem_df = original_df.loc[~original_df['Indicator'].isin(permitted).all(1)]
        cannot_parse= np.unique([location.split("/")[1].strip() for location in rem_df["Location"]])
        print("Cannot find these countries/location -> ",cannot_parse)
    
    file_name = df_name +"_labelled_by_"+Label+".csv"
    
    good_df.to_csv(file_name,index=False)
    return good_df

def change_seq(sequence):
    # Helper function for change_and_check_sequences
    # Random IUPAC codes Interpretation Done here
    # Also remove Gaps or Other letters not in IUPAC codes
    
    change_map = {}
    change_map["R"] = ["A","G"]
    change_map["Y"] = ["C","T"]
    change_map["S"] = ["C","G"]
    change_map["W"] = ["A","T"]
    change_map["K"] = ["G","T"]
    change_map["M"] = ["A","C"]
    change_map["B"] = ["C","T","G"]
    change_map["D"] = ["A","T","G"]
    change_map["H"] = ["C","T","A"]    
    change_map["V"] = ["C","A","G"]
    change_map["N"] = ["A","C","T","G"]
    keys = change_map.keys()
    
    # seq = upper(sequence)
    seq = sequence.upper()
    seq = seq.replace("-","") #Gap removal
    
    mut_seq = [c for c in seq]
    
    for i in range(len(seq)):
        if(seq[i] in keys):
            mut_seq[i] = random.choice(change_map[seq[i]])
    unique = np.unique(mut_seq)
    permitted_list = ['A','C','G','T']
    if(len(unique) != 4):
        will_be_omitted = [x for x in unique if x not in permitted_list]
        print(will_be_omitted)
        for one_by_one in will_be_omitted:
            mut_seq = list(filter((one_by_one).__ne__, mut_seq))
    

    n_sequence = ''.join(mut_seq)
    return n_sequence

def change_and_check_sequences(sequences):
    #Interpret the Sequences and the check it
    print("Interpreting Sequences and Checking")
    n_sequence = []
    for seq in sequences:

        s = set(seq)
        if(len(s)!=4):
            n_seq = change_seq(seq)
            n_sequence.append(n_seq)
        else:
            n_sequence.append(seq)

    for seq in n_sequence:
        s = set(seq)
        if(len(s)!=4):
            print("Not Changed yet!",len(s))
            exit(1)

    return n_sequence

def csv_preparation(id_seq_map,info_file,label_method):
    ## Takes the (Accession ID,sequence) dictionary, the info_file name and the label name as input
    # Interprets All symbols in 4 Neucleotide Symbols
    # Creates All, Train and Test CSV files labeled by the label method provided
    # Doesn't return anything
    cols = ["Accession ID","Virus name","Location","Collection date"]
    ext_  = info_file.split(".")[1]
    if(ext_ == "csv"):
        info_df = pd.read_csv(info_file)
    elif(ext_ == "xls"):
        info_df = pd.read_excel(info_file)
    else:
        print("Wrong file type")
        exit(1)

    ACC_ID_list = list(id_seq_map.keys())
    usable_info_df = info_df.loc[info_df["Accession ID"].isin(ACC_ID_list)] #Extracting the info's of required ACC ID's
    usable_info_df = usable_info_df[cols]

    usable_info_df['Sequence'] = usable_info_df['Accession ID'].map(id_seq_map) # Embedding Sequences in the csv file
    Sequences = np.array(usable_info_df['Sequence'])
    usable_info_df['Sequence'] = change_and_check_sequences(Sequences.copy()) # Interpreting the symbols
    
    print("Labelling Whole Set")
    All_df =  GisAID_Processing("All",usable_info_df,label_method) #Preparing the Whole Set

    if(old_flag == True):
        # If we want to use the original train, test set as set in sample description and used for previous training, testing
        Train_test_info = pd.read_csv("../Sample_description.csv")
        Train_Accessions = list(Train_test_info.loc[Train_test_info["Used for"]=="Training"]["Accession ID"])
        # print(len(Train_Accessions))
        Test_Accessions = Train_test_info.loc[Train_test_info["Used for"]=="Testing"]["Accession ID"]
        Train_df = usable_info_df.loc[usable_info_df["Accession ID"].isin(Train_Accessions)]
        Test_df = usable_info_df.loc[usable_info_df["Accession ID"].isin(Test_Accessions)]
        print("Labelling Training Set")
        GisAID_Processing("Train",Train_df,label_method)
        print("Labelling Testing Set")
        GisAID_Processing("Test",Test_df,label_method)
    else:
        # If we want to create new train, test set.
        y = All_df["Indicator"]
        train, test = train_test_split(All_df, test_size=0.2,stratify=y)
        print("Labelling Training Set")
        GisAID_Processing("Train",train,label_method)
        print("Labelling Testing Set")
        GisAID_Processing("Test",test,label_method)

def main():
    parser = argparse.ArgumentParser(description="Input Processing\n,Requires the sequence fasta file and the Information File")
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--input','-i',help='read sequence data from fasta file', required=True, type=str)
    required.add_argument('--info_file','-if',help='info file for accession ID, sample file-> gisaid_cov2020_acknowledgement_table.csv',required=True, type=str)
    required.add_argument('--label','-l',help='Label on which parameter (Death/CFR_confirmed_cases/CFR_Recovery/CFR_Infrastructure)',required=True,type=str)

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--old','-o',help='Use Old Train/Test Accession ID',type=int)
    pars = parser.parse_args()
    input_fasta = pars.input
    info_file = pars.info_file
    label = pars.label
    # print(pars.old)
    if (pars.old == 1):
        # print("Here")
        global old_flag
        old_flag = True

    print(input_fasta,info_file)
    id_seq_map = read_fasta(input_fasta)
    csv_preparation(id_seq_map,info_file,label)
    


if __name__ == "__main__":
    main()