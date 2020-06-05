import pandas as pd
import itertools
import numpy as np
import os


def gap_features(df, nucleotides, max_gap = 30):
    ret_def = pd.DataFrame()
    for i in range(1, max_gap + 1):
        for x1 in nucleotides:
            for x2 in nucleotides:
                col_name = 'GAP_' + x1 + '_' + str(i) + '_' + x2
                print('Generating Feature ' + col_name)
                ret_def[col_name] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)
                idx = 0
                for RNA in df['Sequence']:
                    cnt = 0
                    for j in range(len(RNA) - (i + 1)):
                        if RNA[j] == x1 and RNA[j + i + 1] == x2:
                            cnt += 1
                    ret_def[col_name].at[idx] = np.int32(cnt)
                    idx += 1
    return ret_def


def position_independent(df, order, nucleotides, other_nucleotides):
    ret_def = pd.DataFrame()
    for ord_ in range(1, order + 1):
        for p in itertools.product(nucleotides, repeat=ord_):
            p = ''.join(p)
            ret_def[p] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)
            print('Generating Feature ' + p)
            idx = 0
            for RNA in df['Sequence']:
                cnt = RNA.count(p)
                ret_def[p].at[idx] = np.int16(cnt)
                idx += 1
    for p in other_nucleotides:
        ret_def[p] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)
        print('Generating Feature ' + p)
        idx = 0
        for RNA in df['Sequence']:
            cnt = RNA.count(p)
            ret_def[p].at[idx] = np.int32(cnt)
            idx += 1

    print(ret_def.shape)
    return ret_def


def position_specific(df, order, nucleotides, max_length=30000):
    subseq_list = itertools.product(nucleotides, repeat=order)
    count = 0
    subseq_dict = {}
    ret_def = pd.DataFrame()
    num_features = max_length // order
    print(num_features)

    for p in subseq_list:
        p = ''.join(p)
        count += 1
        subseq_dict[p] = count

    for i in range(num_features):
        p = 'pos_' + str(order*i) + '_' + str(order*(i+1)-1)
        ret_def[p] = pd.Series(data=(df.shape[0] * [0])).astype(np.int32)

    idx = 0
    total_rna = df.shape[0]

    for RNA in df['Sequence']:
        length = len(RNA)
        for i in range(0, min(length, max_length), order):
            substr = RNA[i:i+order]
            p = 'pos_' + str(i) + '_' + str(i + order - 1)
            if substr in subseq_dict:
                ret_def[p].at[idx] = np.int32(subseq_dict[substr])
        idx += 1
        print('Generated feature for ',idx,'/',total_rna,' RNAs')

    return ret_def


def extract_featutres(filename, outdir=''):
    nucleotides_ = ['A', 'C', 'T', 'G']
    #iupac_neucleotides = ['R', 'Y', 'K', 'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N', '-']
    iupac_neucleotides = []

    dir = 'Features/' + outdir + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    filepath = '../Input/' + filename + '.csv'

    df = pd.read_csv(filepath, delimiter=',')
    df['Sequence'] = df['Sequence'].str.upper()

    print('Data Shape: ', df.shape)

    if 'Indicator' in df.columns:
        labels = pd.DataFrame(df['Indicator'].astype(np.int8), columns=['Indicator'])
        labels.to_hdf( dir + filename + '_labels.h5', key='labels')

    df_pos_ind = position_independent(df, 4, nucleotides_, iupac_neucleotides).astype(np.int32)
    df_pos_ind.to_hdf(dir + filename + '_pi.h5', key='pi')

    df_pos_ps = position_specific(df, 5, nucleotides_).astype(np.int32)
    df_pos_ps.to_hdf(dir + filename + '_ps.h5', key='ps')

    df_gap = gap_features(df, nucleotides_).astype(np.int32)
    df_gap.to_hdf(dir + filename + '_gap.h5', key='gap')


if __name__ == "__main__":
    filename = input('Filename:')
    extract_featutres(filename)