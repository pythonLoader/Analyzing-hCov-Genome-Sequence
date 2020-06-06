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
    cout << "\n\nDistance calculation file starts running.....\n";

    string basePath = "";

    vector<string> fileNames, filesWithPath, taxaNames;

    string input = "input.maw.fasta";

    vector<Set> maws;
    vector<string> taxas;

    cout << "Extracting MAWs from files....\n";

    auto start = cloc::now();

    maws = GetAllSetsFromOneFile(input,taxas);

    sec duration = cloc::now() - start;

    cout << "MAWs extracted. Time taken (sec)--> " << duration.count() << endl << endl;

    int num_of_genes = maws.size();


    cout << num_of_genes << " species found\n\n";

    cout << "Calculating distance matrix.....\n\n";

    int distMethod = stoi(string(argv[1]));

    start = cloc::now();
    CalculateDistanceMatrix(maws, diffMatrix, num_of_genes, distMethod);
    duration = cloc::now() - start;

    cout << "Distance matrix calucalted. Time taken (sec)--> " << duration.count() << endl;

    printMatrix(basePath, taxas, diffMatrix, num_of_genes, distMethod, CSV);

    cout << "Distance matrix printed...\n\n ";

    return 0;
}