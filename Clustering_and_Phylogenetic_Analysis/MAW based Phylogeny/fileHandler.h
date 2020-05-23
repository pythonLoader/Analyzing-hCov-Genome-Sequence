#pragma once

#include <bits/stdc++.h>
#include <boost/filesystem.hpp>

#include "set.h"
#include "defs.h"

using namespace std;
using namespace boost::filesystem;

void GetAllFileName(string pathName, vector<string> &fileNames)
{
    path p(pathName);

    for (auto i = directory_iterator(p); i != directory_iterator(); i++)
    {
        if (!is_directory(i->path())) //ignoring directories
        {
            fileNames.push_back(i->path().filename().string());
            //cout<<i->path().filename().string()<<endl;
        }
    }
}

string TaxaNameFromFileName(string fileName)
{
    string s = fileName;
    string delimiter = ".";
    string token = s.substr(0, s.find(delimiter));
    return token;
}

vector<Set> GetAllSetsFromMultipleFiles(vector<string> fileNames, vector<string> baseNames, vector<string> &taxas)
{
    Set emptySet;
    vector<Set> mawset;
    for (int i = 0; i < fileNames.size(); i++)
    {
        mawset.push_back(emptySet);
        mawset[i].MakeFromFile(fileNames[i]);
        mawset[i].taxaName = TaxaNameFromFileName(baseNames[i]);
        taxas.push_back(TaxaNameFromFileName(baseNames[i]));
        mawset[i].SortElements();
    }

    return mawset;
}

vector<Set> GetAllSetsFromOneFile(string fileName, vector<string> &taxas)
{
    ifstream file(fileName);
    vector<Set> sets;
    string line;

    Set set, emptySet;

    while (getline(file, line))
    {
        if (line[0] == '>') //line with taxa names as in FASTA format
        {
            line.erase(line.begin());
            set = emptySet;
            istringstream ss(line);
            //ss>>line; // first word taken as taxa name 
            set.SetTaxaName(line);
            taxas.push_back(line);
            sets.push_back(set);
        }
        else if (line.size() > 0)
        {
            sets.back().AddElements(line);
        }
    }

    return sets;
}