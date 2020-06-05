import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import pickle as pkl
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def draw_roc_curve(test_y, prob_y, dir):
    fpr, tpr, _ = roc_curve(test_y, prob_y)
    roc_auc = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='LightGBM Classifier')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (LightGBM Classifier)')

    plt.legend(loc="lower right")
    plt.savefig(dir + 'AUROC.png')
    plt.show()


def train(train_x, train_y):
    extraTree = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=1)
    clf = LGBMClassifier(objective='binary')

    steps = [('SFM', SelectFromModel(estimator=extraTree)),
             ('Scaler', StandardScaler()),
             ('CLF', clf)]

    pipeline = Pipeline(steps)

    pipeline.fit(train_x, train_y)

    return pipeline


def score(test_y, pred_y, prob_y):
    accuracy = metrics.accuracy_score(test_y, pred_y)
    precision = metrics.precision_score(test_y, pred_y)
    recall = metrics.recall_score(test_y, pred_y)
    f1 = metrics.f1_score(test_y, pred_y)
    roc = metrics.roc_auc_score(test_y,prob_y[:,1])
    return accuracy, precision, recall, f1, roc


def k_fold_cv(data_x , data_y, outdir, folds=10):
    dir = 'Results/' + outdir + '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    print(data_x.shape)
    print(data_y.shape)

    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    skf.get_n_splits(data_x, data_y)
    i = 0
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    rocs = []

    for train_index, test_index in skf.split(data_x, data_y):
        i += 1

        train_x, test_x = data_x.ix[train_index], data_x.ix[test_index]
        train_y, test_y = data_y.ix[train_index], data_y.ix[test_index]

        pipeline = train(train_x, train_y)

        pred_y = pd.DataFrame(pipeline.predict(test_x), columns=['Prediction'])
        prob_y = pd.DataFrame(pipeline.predict_proba(test_x)).values

        acc, pre, re, f1, roc = score(test_y, pred_y, prob_y)
        accuracies.append(acc)
        precisions.append(pre)
        recalls.append(re)
        f1_scores.append(f1)
        rocs.append(roc)

        print('Fold: ',i)
        print(acc, pre, re, f1, roc)

    accuracies = pd.DataFrame(accuracies, columns=['Accuracy'])
    precisions = pd.DataFrame(precisions, columns=['Precision'])
    recalls = pd.DataFrame(recalls, columns=['Recall'])
    f1_scores = pd.DataFrame(f1_scores, columns=['F1 Scores'])
    rocs = pd.DataFrame(rocs, columns=['AUROC'])
    scores = pd.concat([accuracies,precisions,recalls, f1_scores, rocs], axis=1, sort=False)

    scores.to_csv(dir + 'Cross Validation Result.csv', index=False)


def load_data(filename):
    pi = pd.DataFrame(pd.read_hdf(filename + '_pi.h5', key='pi')).astype(np.int32)
    gap = pd.DataFrame(pd.read_hdf(filename + '_gap.h5', key='gap')).astype(np.int32)
    ps = pd.DataFrame(pd.read_hdf(filename + '_ps.h5', key='ps')).astype(np.int32)

    data_x = pd.concat([pi, gap, ps], axis=1, sort=False).astype(np.int32)
    data_y = pd.DataFrame(pd.read_hdf(filename + '_labels.h5', key='labels')).astype(np.int32)
    return data_x, data_y


def join_train_test(train_filename, test_filename, outdir):
    train_filename = 'Features/' + outdir + '/' + train_filename
    test_filename = 'Features/' + outdir + '/' + test_filename

    train_x, train_y = load_data(train_filename)
    test_x, test_y = load_data(test_filename)
    data_x = pd.concat(train_x, test_x, ignore_index = True)
    data_y = pd.concat(train_y, test_y, ignore_index=True)
    return data_x, data_y

def save_results(dir,acc,pre,re,f1,roc):
    file = open(dir + "Results.txt", "w")
    file.write('Accuracy: ' + str(acc) + '\n')
    file.write('Precision: ' + str(pre) + '\n')
    file.write('Recall: ' + str(re) + '\n')
    file.write('F1 Score: ' + str(f1) + '\n')
    file.write('AUROC Score: ' + str(roc) + '\n')
    file.close()


def split_train(train_filename, test_filename, outdir):
    train_filename = 'Features/' + outdir + '/' + train_filename
    test_filename = 'Features/' + outdir + '/' + test_filename

    train_x, train_y = load_data(train_filename)
    test_x, test_y = load_data(test_filename)

    dir = 'Results/' + outdir + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    pipeline = train(train_x, train_y)
    f = open(dir + 'Model.pkl', 'wb')
    pkl.dump(pipeline, f)

    pred_y = pd.DataFrame(pipeline.predict(test_x), columns=['Prediction'])
    prob_y = pd.DataFrame(pipeline.predict_proba(test_x)).values
    acc, pre, re, f1, roc = score(test_y, pred_y, prob_y)

    prob_y = prob_y[:,1]
    draw_roc_curve(test_y, prob_y, dir)
    prob_y = pd.DataFrame(prob_y, columns=['Probability of 1'])
    test_y = pd.DataFrame(test_y.values, columns=['Indicator'])
    result = pd.concat([test_y, pred_y, prob_y], axis=1, sort=False)
    result.to_csv(dir + 'Prediction.csv', index=False)

    print('Accuracy: ', acc)
    print('Precision: ', pre)
    print('Recall: ', re)
    print('F1 Score: ', f1)
    print('ROC AUC Score', roc)
    save_results(dir, acc, pre, re, f1, roc)


if __name__ == "__main__":
    filename = 'All_Data'
