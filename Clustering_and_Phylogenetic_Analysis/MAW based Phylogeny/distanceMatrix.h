#include <bits/stdc++.h>
#include <boost/filesystem.hpp>

#include "set.h"
#include "fileHandler.h"
#include "defs.h"

using namespace std;

Set GetGlobalUnion(vector<Set> &mawset, int num_of_genes)
{
    Set uset;

    for (int i = 0; i < num_of_genes; i++)
        uset = uset.Union(mawset[i]);

    return uset;
}

int CalculateDistanceMatrix(vector<Set> &mawset, double diffMatrix[][MAX_NUM_GENE], int num_of_genes, int diffIndex)
{

    Set uSet, iSet, sdSet, a, b, c, globalSet;

    if (diffIndex == MAW_SMD)
        globalSet = GetGlobalUnion(mawset, num_of_genes);

    for (int i = 0; i < num_of_genes; i++)
    {
        //cout<<i<<" th gene"<<endl;
        for (int j = 0; j < i; j++)
        {
            
            switch (diffIndex)
            {
            case MAW_JACCARD:

                uSet = mawset[i].Union(mawset[j]);
                iSet = mawset[i].Intersection(mawset[j]);

                diffMatrix[i][j] = diffMatrix[j][i] = 1.0 - (1.0 * iSet.Cardinality() / uSet.Cardinality());

                break;

            case MAW_LWI_SD:

                sdSet = mawset[i].SymmetricDifference(mawset[j]);

                diffMatrix[i][j] = diffMatrix[j][i] = sdSet.LengthWeightedIndex();

                break;

            case MAW_OVERLAP_CO:
            {
                iSet = mawset[i].Intersection(mawset[j]);
                int minCardinality = min(mawset[i].Cardinality(), mawset[j].Cardinality());

                diffMatrix[i][j] = diffMatrix[j][i] = 1 - (1.0 * iSet.Cardinality() / minCardinality);

                break;
            }

            case MAW_SMD:

                iSet = mawset[i].Intersection(mawset[j]);

                diffMatrix[i][j] = diffMatrix[j][i] = 1 - (1.0 * iSet.Cardinality() / globalSet.Cardinality());

                break;

            case MAW_SORENSEN_DICE:
            {
                iSet = mawset[i].Intersection(mawset[j]);

                diffMatrix[i][j] = diffMatrix[j][i] = 1 - (2.0 * iSet.Cardinality() / (mawset[i].Cardinality() + mawset[j].Cardinality()));

                break;
            }
            }
        }
    }
}

void printMatrix(string outputPath, vector<string> taxas, double diffMatrix[][MAX_NUM_GENE], int num_of_genes, int distIndex, int formatIndex)
{

    switch (formatIndex) //diferent output format for different file format
    {
    case CSV: // CSV dist matrix used as input to Dendropy's newick tree file generation
    {
        string outFile = outputPath + "/distanceMatrix" + to_string(distIndex) + ".csv";
        ofstream out;
        out.open(outFile);

        for (int i = 0; i < num_of_genes; i++)
            out << "," << taxas[i];
        out << endl;

        for (int i = 0; i < num_of_genes; i++)
        {
            out << taxas[i];

            for (int j = 0; j < num_of_genes; j++)
                out << "," << diffMatrix[i][j];

            out << endl;
        }

        if (out.is_open())
            out.close();

        break;
    }

    case PHYLIP:
    {

        string outFile = outputPath + "/distanceMatrix.phy";
        ofstream out;
        out.open(outFile);

        out << num_of_genes << endl;

        for (int i = 0; i < num_of_genes; i++)
        {
            out << taxas[i];

            for (int j = 0; j < num_of_genes; j++)
                out << "\t" << diffMatrix[i][j];

            out << endl;
        }

        if (out.is_open())
            out.close();

        break;
    }

    case TSV:
    {

        string outFile = outputPath + "/distanceMatrix" + to_string(distIndex) + ".csv";
        ofstream out;
        out.open(outFile);

        for (int i = 0; i < num_of_genes; i++)
            for (int j = i + 1; j < num_of_genes; j++)
                out << taxas[i] << "\t" << taxas[j] << "\t" << diffMatrix[i][j] << endl;

        if (out.is_open())
            out.close();

        break;
    }
    }
}


