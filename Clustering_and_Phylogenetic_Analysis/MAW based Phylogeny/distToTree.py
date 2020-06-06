import dendropy
import sys

distFile= 'out_dist'+sys.argv[1]+'.csv'
njtreeFile= 'out_tree_NJ'+ sys.argv[1]+'.txt'

pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
        src=open(distFile),
        delimiter=",")

nj_tree = pdm.nj_tree()

f = open(njtreeFile, "w")
f.write(nj_tree.as_string("newick"))

# upgtreeFile= 'newickUPG'+ sys.argv[1]+'.txt'
# upg_tree = pdm.upgma_tree()
# f2 = open(upgtreeFile, "w")
# f2.write(nj_tree.as_string("newick"))

print("Trees printed")