## This repository is the official implementation of the Analyzing-hCov-genome-sequence pipeline. 

If you use any part of this repository, we shall be very grateful if you cite our paper [Analyzing hCov genome sequences: Applying Machine Intelligence and beyond](https://www.biorxiv.org/content/10.1101/2020.06.03.132506v1)

# Usage

## Installation 
1. Please install [Tensorflow version: 2.2.0](hhttps://pypi.org/project/tensorflow/2.2.0/). (Other 2.x versions should work, but have not been tested. Use gpu for better performance)
2. Please install [Keras version: 2.3.1](https://pypi.org/project/Keras/2.3.1/).
3. Other python libraries used for this project can be installed by running the following command
> pip install -r [requirements.txt](https://github.com/pythonLoader/Analyzing-hCov-Genome-Sequence/blob/master/requirements.txt) 

## Input Preprocessing and Labeling

1. Keep the input sequence fasta file (file used in this analysis can be downloaded from [here](https://drive.google.com/file/d/1ZSOXIY_ifGbQuq3AsmZWhhEmRw3nNncm/view) ) and the info file ([sample file](https://github.com/pythonLoader/Analyzing-hCov-Genome-Sequence/blob/master/Input/gisaid_cov2020_acknowledgement_table.csv) present in the input directory) in the Input directory
2. The input_processing.py script handles the input processing and labelling task. It requires 3 mandatory parameters and 1 optional parameters. They are:
    - input
    - info_file
    - label
    - old
3. There are **four** options for labelling:
    - Death
    - CFR_Recovery
    - CFR_confirmed_cases
    - CFR_Infrastructure
4. Set old parameter to **1** to use the old Training/Testing Accession ID's for input preprocessing and labelling. Don't use this parameter for generating new Training/Testing set. The models are pre-trained with Death Labelling. 
5. Sample command:
> python input_processing.py --input <input_fasta_file_name> --info_file <info_file_name> --label <label_option> --old 1(optional)

## Country-wise Representative Sequence Identification and Phylogenetic Analysis

1. Navigate to the [<Clustering_and_Phylogenetic_Analysis>](https://github.com/pythonLoader/Analyzing-hCov-Genome-Sequence/tree/master/Clustering_and_Phylogenetic_Analysis) directory.
2. The controller.py script is a one-stop service for all of related analysis. It requires 2 parameters:
    - label (Same as the Input Folder Options)
    - method
3. There are **4** options for method:

    |Method | Description|
    | ------------ |:----------:|
    |Euclidean | Simple Euclidean distance-based method among the 3-mers of the genome sequence |
    |Novel_Fast_Vector| 18-dimensional Novel Fast Vector Sequence Comparison Analysis |
    |Accumulated_Fast_Vector| 18-dimenasional Accumulated Fast Vector Sequence Comparison Analysis |
    |MAW| Minimum Absent Word Analysis|
4. Only MAW requires additional **4** parameters:
    - Minimum_MAW_Length
    - Maximum_MAW_Length
    - Distance_Method
    - Fasta File Name (Must be kept in the Input Directory)
5. Sample Command:
> python controller.py --label <label_option> --method <method_option>
