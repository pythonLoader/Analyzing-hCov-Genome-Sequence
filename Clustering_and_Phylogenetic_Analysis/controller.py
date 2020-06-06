import subprocess
import argparse
import re
import os

def main():
    #------------------------------------ Parsing Routines --------------------------------------------------#
    methods = 	('Euclidean','cosine','Novel_Fast_Vector','Accumulated_Fast_Vector',"MAW")
    labels = ('Death','CFR_confirmed_cases','CFR_Recovery','CFR_Infrastructure')
    parser = argparse.ArgumentParser(description="genome sequence set splitter by country")
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--label','-l',choices=labels, help='Label on which parameter (Death/CFR_confirmed_cases/CFR_Recovery/CFR_Infrastructure)',required=True,type=str)
    required.add_argument('--method','-m',type=str, required=True, choices = methods,help='Available methods: ' +"'"+"', '".join(methods) +"'")
    
    required_by_methods = parser.add_argument_group('Arguments required by some methods')
    required_by_methods.add_argument('--Minimum_MAW_Length', '-min_ml', help="Minimum length of MAWs", type=str)
    required_by_methods.add_argument('--Maximum_MAW_Length', '-max_ml', help="Maximum length of MAWs",type=str)
    required_by_methods.add_argument('--Distance_Method', '-dm', help="Distance Method (e.g 1 for Jaccard Distance, see more on defs.h)",type=str)
    required_by_methods.add_argument('--fasta_file', '-ff', help="Fasta_File_Name",type=str)
    

    post_processing_flag = False
    try:
        pars = parser.parse_args()
        label = pars.label
        method = pars.method

        # Handling MAW parameters input
        if (not pars.Minimum_MAW_Length and ('MAW' in pars.method)): 
            parser.error( "method MAW requires option -min_ml/--Minimum_MAW_Length to be set")
        if (not pars.Maximum_MAW_Length and ('MAW' in pars.method)): 
            parser.error( "method MAW requires option -max_ml/--Maximum_MAW_Length to be set")
        if (not pars.Distance_Method and ('MAW' in pars.method)):
             parser.error( "method MAW requires option -dm/--Distance_Method to be set")
        if (not pars.fasta_file and ('MAW' in pars.method)): 
            parser.error( "method MAW requires option -ff/--fasta_file to be set")

    except IOError:
        # args.error(str(msg))
        print("IOerror")
    #------------------------------------ Parsing Routines --------------------------------------------------#


    if(method != "MAW"):

        post_processing_flag = False
        subprocess.call(['python','file_splitter.py','-l',label])
   
    if(method == "Novel_Fast_Vector"):
        subprocess.call((['python','Novel_Fast_Vector.py'])) # Novel Fast Vector Distance Matrix Generation Routine
        post_processing_flag = True
    elif(method == "Accumulated_Fast_Vector"):
        subprocess.call(['python','Accumulated_Natural_Vector.py']) # Accumulated Fast Vector Distance Matrix Generation Routine
        post_processing_flag=True
    elif(method == "Euclidean"):
        subprocess.call(['python','Euclidean.py']) # Euclidean Distance Mat Generation Routine
        post_processing_flag=True
    
    elif(method == "MAW"):
        min_maw_length = pars.Minimum_MAW_Length
        max_maw_length = pars.Maximum_MAW_Length
        distance_method = pars.Distance_Method
        fasta_file = pars.fasta_file

        required_file = "../Input/" + fasta_file
        fasta_file_path = os.path.abspath(required_file)
        subprocess.call(['./runMAW.sh','dist',min_maw_length, max_maw_length, distance_method,fasta_file_path])
        

    if(post_processing_flag == True):
        subprocess.call(['python','representative_sequence_finder.py'])
        subprocess.call(['python','All_Country_Distance_Matrix_generator.py','-l',label,'-m',method])
        #-----------------Tree Generation------------------------#
        subprocess.call(['python','distToNewick.py','-m',method])
        tree_file = open("tree_nj_"+method+".txt","r")
        stri = tree_file.read()
        print("Initial->",stri)
        tree_file.close()
        tree_file = open("tree_nj_"+method+".txt", 'w')
        stri_ = stri[stri.find('('):] #need to remove [&U] in front for dendroscope operation
        print(stri_)
        tree_file.write(stri_)
        tree_file.close()
        #-----------------Tree Generation------------------------#
    else:
        distance_method = pars.Distance_Method
        subprocess.call(['./runMAW.sh','tree',distance_method])

if __name__ == "__main__":
    main()