1. Create a working folder in this directory and put a "maws" folder in it.
2. Get the MAWs from sequences and put it in the maws folder. (see this https://github.com/solonas13/maw)
3. Now run the script 
        ./run.sh <working_folder_name> <distance_method>

   For example, in current scenario, it would be:
        ./run.sh data 1
 
   See defs.h file for distance method codes (e.g. 1 for Jaccard index)

4. The output files (distance matrix and tree in newick format) will be printed in the working folder.


Requirements:
1. Unix OS (for MAW generation part)
2. DendroPy (python library - used for producing trees)
3. Boost library for C++
