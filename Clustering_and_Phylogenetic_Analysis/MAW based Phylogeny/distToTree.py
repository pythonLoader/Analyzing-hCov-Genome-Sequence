import dendropy
import sys

distFile= sys.argv[1] + '/distanceMatrix'+sys.argv[2]+'.csv'
njtreeFile=  sys.argv[1] + '/newickNJ'+ sys.argv[2]+'.txt'
upgtreeFile=  sys.argv[1] + '/newickUPGMA'+ sys.argv[2]+'.txt'

pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
        src=open(distFile),
        delimiter=",")
nj_tree = pdm.nj_tree()
upg_tree = pdm.upgma_tree()

f = open(njtreeFile, "w")
f.write(nj_tree.as_string("newick"))

f2 = open(upgtreeFile, "w")
f2.write(nj_tree.as_string("newick"))

print("Trees printed")