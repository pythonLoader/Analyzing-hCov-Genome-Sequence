import dendropy
import sys
import argparse

def main():

        methods = ('Euclidean','cosine','Novel_Fast_Vector','Accumulated_Fast_Vector')
        parser = argparse.ArgumentParser(description="Final Distance Matrix Generator")
        required = parser.add_argument_group('Required Arguments')
        required.add_argument('--method','-m',type=str, required=True, choices = methods,help='Available methods: ' +"'"+"', '".join(methods) +"'")
        pars = parser.parse_args()
        method = pars.method
        distFile= 'Final_distance_matrix_'+method+'.csv'
        treeFile= 'tree_nj_'+method+'.txt'

        pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
                src=open(distFile),
                delimiter=",")
        nj_tree = pdm.nj_tree()
        # UPG_tree = pdm.upgma_tree()
        f = open(treeFile, "w")
        f.write(nj_tree.as_string("newick"))

if __name__ == "__main__":
    main()