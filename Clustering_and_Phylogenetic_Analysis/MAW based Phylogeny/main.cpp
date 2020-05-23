#include <bits/stdc++.h>
#include <chrono>

#include "distanceMatrix.h"
#include "fileHandler.h"
#include "set.h"
#include "defs.h"

using namespace std;
using cloc = chrono::system_clock;
using sec = chrono::duration<double>;

double diffMatrix[MAX_NUM_GENE][MAX_NUM_GENE];

int main(int argc, char *argv[])
{
    cout << "Distance calculation file starts running.....\n";

    string folder(argv[1]);
    string basePath = folder;
    string pathName = basePath + "/maws";
    vector<string> fileNames, filesWithPath, taxaNames;

    GetAllFileName(pathName, fileNames);

    for (int i = 0; i < fileNames.size(); i++)
    {
        string str = pathName + "/" + fileNames[i];
        filesWithPath.push_back(str);
        //cout<<filesWithPath[i]<<endl;
    }

    vector<Set> maws;
    vector<string> taxas;

    cout << "Extracting MAWs from files....\n";

    auto start = cloc::now();

    if (fileNames.size() > 1)
        maws = GetAllSetsFromMultipleFiles(filesWithPath, fileNames, taxas); //multiple file format
    else
        maws = GetAllSetsFromOneFile(filesWithPath[0], taxas); //single file format

    sec duration = cloc::now() - start;

    cout << "MAWs extracted. Time taken (sec)--> " << duration.count() << endl
         << endl;

    int num_of_genes = maws.size();

    //for (int i = 0; i < taxas.size(); i++)
    //{
    //    cout<<taxas[i]<<endl;
    //}

    cout << num_of_genes << " species found\n\n";

    //double diffMatrix[MAX_NUM_GENE][MAX_NUM_GENE];

    cout << "Calculating distance matrix.....\n\n";

    int distMethod = stoi(string(argv[2]));

    start = cloc::now();
    CalculateDistanceMatrix(maws, diffMatrix, num_of_genes, distMethod);
    duration = cloc::now() - start;

    cout << "Distance matrix calucalted. Time taken (sec)--> " << duration.count() << endl;

    printMatrix(basePath, taxas, diffMatrix, num_of_genes, distMethod, CSV);

    cout << "Distance matrix printed...\n\n " << endl;

    return 0;
}