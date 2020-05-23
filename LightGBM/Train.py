import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def draw_roc_curve(test_y, prob_y):
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
    plt.savefig('Figures/LGBM.png')
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
    roc = metrics.roc_auc_score(test_y,prob_y[:,1])
    return accuracy, precision, recall, roc


def k_fold_cv(data_x, data_y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    skf.get_n_splits(data_x, data_y)
    i = 0
    accuracies = []
    precisions = []
    recalls = []
    rocs = []

    for train_index, test_index in skf.split(data_x, data_y):
        i += 1
        train_x, test_x = data_x.ix[train_index], data_x.ix[test_index]
        train_y, test_y = data_y.ix[train_index], data_y.ix[test_index]
        pipeline = train(train_x, train_y)
        pred_y = pd.DataFrame(pipeline.predict(test_x), columns=['Prediction'])
        prob_y = pd.DataFrame(pipeline.predict_proba(test_x)).values
        acc, pre, re, roc = score(test_y, pred_y, prob_y)
        accuracies.append(acc)
        precisions.append(pre)
        recalls.append(re)
        rocs.append(roc)
        print('Fold: ',i)
        print(acc, pre, re, roc)

    accuracies = pd.DataFrame(accuracies, columns=['Accuracy'])
    precisions = pd.DataFrame(precisions, columns=['Precision'])
    recalls = pd.DataFrame(recalls, columns=['Recall'])
    rocs = pd.DataFrame(rocs, columns=['ROC'])
    scores = pd.concat([accuracies,precisions,recalls,rocs], axis=1, sort=False)
    scores.to_csv('Data/Result/Cross Validation Result 2.csv', index=False)


def load_data(filename='Independent 1'):
    pi = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_pi.h5', key='pi')).astype(np.int32)
    gap = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_gap.h5', key='gap')).astype(np.int32)
    ps = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_ps.h5', key='ps')).astype(np.int32)

    data_x = pd.concat([pi, gap, ps], axis=1, sort=False).astype(np.int32)
    data_y = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_labels.h5', key='labels')).astype(np.int32)
    return data_x, data_y


def split_train(data_x, data_y):
    train_x = data_x.ix[:8109]
    train_y = data_y.ix[:8109]

    test_y = data_y.ix[8109:]
    test_x = data_x.ix[8109:]

    # test_x, test_y = load_data('Independent 1')

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    pipeline = train(train_x, train_y)
    pred_y = pd.DataFrame(pipeline.predict(test_x), columns=['Prediction'])
    prob_y = pd.DataFrame(pipeline.predict_proba(test_x)).values
    acc, pre, re, roc = score(test_y, pred_y, prob_y)

    prob_y = prob_y[:,1]
    draw_roc_curve(test_y, prob_y)
    prob_y = pd.DataFrame(prob_y, columns=['Probability of 1'])
    test_y = pd.DataFrame(test_y.values, columns=['Indicator'])
    result = pd.concat([test_y, pred_y, prob_y], axis=1, sort=False)
    result.to_csv('Data/Result/Indicator/LGB/lgb_pred_prob.csv', index=False)

    print('Accuracy: ', acc)
    print('Precision: ', pre)
    print('Recall: ', re)
    print('ROC AUC Score', roc)


if __name__ == "__main__":
    filename = 'All_Data'
    classifier = 'LGB'

    pi = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_pi.h5', key='pi')).astype(np.int32)
    gap = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_gap.h5', key='gap')).astype(np.int32)
    ps = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_ps.h5', key='ps')).astype(np.int32)

    data_x = pd.concat([pi,gap,ps], axis=1, sort=False).astype(np.int32)
    data_y = pd.DataFrame(pd.read_hdf('Data/Features/' + filename + '_labels.h5', key='labels')).astype(np.int32)
    #k_fold_cv(data_x, data_y)
    split_train(data_x, data_y)
