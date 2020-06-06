# Instructions:

1. You need to have the MAW suite installed on your directory. This suite is taken from https://github.com/solonas13/maw
   You will be prompted to install MAW from the controller scripts. Required files for MAW will be fetched.


2. To produce MAW based phylogeny results, you need to set some parameters.

     a. Minimum MAW length
     b. Maximum MAW length
     c. Distance Method (e.g 1 for Jaccard, 2 for Length Weighted Index. See more on defs.h)
     d. Input file name (it expects the input file to be in the Input folder of previous directory)


3. A sample command to run the analysis.

     python controller.py --label Death --method MAW -min_ml 12 -max_ml 15 -dm 1 -ff input.fasta



Requirements:

1. Unix OS (for MAW generation part)
2. DendroPy (python library - used for producing trees)




