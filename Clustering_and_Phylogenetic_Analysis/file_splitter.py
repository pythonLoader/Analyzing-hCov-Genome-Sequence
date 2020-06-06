import pandas as pd 
import sys,os
import numpy as np
import argparse

def split_by_country(df):
    locations = df["Location"]

    loc,num = np.unique(locations,return_counts=True)

    countries = []
    for country in loc:
        cont_ = country.split('/')[1].strip()
        if cont_ not in countries:
            countries.append(cont_)
    countries.sort()

    if not os.path.exists('All_Countries_Splitted'):
        os.mkdir('All_Countries_Splitted')

    direc = 'All_Countries_Splitted'
    for cont_ in countries:
        n_df = df.loc[df["Location"].str.contains(cont_)]
        f_name = direc+"/"+cont_+".csv"
        n_df.to_csv(f_name,index=False)



def main():
    parser = argparse.ArgumentParser(description="genome sequence set splitter by country")
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--label','-l',help='Label on which parameter (Death/CFR_confirmed_cases/CFR_Recovery/CFR_Infrastructure)',required=True,type=str)
    pars = parser.parse_args()
    label = pars.label
    # print(label)
    required_file = "../Input/All_labelled_by_" + label +".csv"
    df = pd.read_csv(required_file)
    print(df.shape)
    split_by_country(df.copy())


if __name__ == "__main__":
    main()

