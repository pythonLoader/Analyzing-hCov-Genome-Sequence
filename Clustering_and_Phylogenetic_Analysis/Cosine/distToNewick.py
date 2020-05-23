import dendropy
import sys

distFile= 'Final_distance_matrix.csv'
treeFile= 'tree_nj_cosine.txt'

pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
        src=open(distFile),
        delimiter=",")
nj_tree = pdm.nj_tree()
# UPG_tree = pdm.upgma_tree()
f = open(treeFile, "w")
f.write(nj_tree.as_string("newick"))
