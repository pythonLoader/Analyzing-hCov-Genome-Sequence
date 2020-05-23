#!/bin/bash

# What this script does :

#   a) runs the C++ code to get Distance Matrix from MAW Sets. 
#       1. First argument: The folder containing the MAW 
#       2. Second Argument: The Distance metric to be used (e.g 1 for Jaccard, 2 for LWI on SD)

#   b) uses the Python script to convert Distance Matrix to Neighbour Join Newick Tree Representation

g++ main.cpp -std=c++0x -lboost_system -lboost_filesystem -o main
./main $1 $2

python3 distToTree.py $1 $2
