# -*- coding: utf-8 -*-
import numpy as np
import itertools
import pandas as pd


def calculate_rand_index(labels_true, labels_pred):
    """
    Adapted from:
    https://pml.readthedocs.io/en/latest/_modules/pml/unsupervised/clustering.html#ClusteredDataSet.calculate_rand_index

    Calculate the Rand index, a measurement of quality for the clustering results.  It is
    essentially the percent accuracy of the clustering.

    The clustering is viewed as a series of decisions.  There are
    N*(N-1)/2 pairs of samples in the dataset to be considered.  The
    decision is considered correct if the pairs have the same label and
    are in the same cluster, or have different labels and are in different
    clusters.  The number of correct decisions divided by the total number
    of decisions gives the Rand index, or accuracy.

    Returns:
      The accuracy, a number between 0 and 1.  The closer to 1, the better
      the clustering.

    """

    correct = 0
    total = 0
    for index_combo in itertools.combinations(range(len(labels_true)), 2):
        index1 = index_combo[0]
        index2 = index_combo[1]

        same_class = (labels_true[index1] == labels_true[index2])
        same_cluster = (labels_pred[index1]
                        == labels_pred[index2])

        if same_class and same_cluster:
            correct += 1
        elif not same_class and not same_cluster:
            correct += 1

        total += 1

    return float(correct) / total


# labels_true=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 1, 0, 2, 2, 2, 0]
# labels_pred=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
# print calculate_rand_index(labels_true, labels_pred)

def calculate_purity(labels_true, labels_pred):
    y_actu = pd.Series(labels_true, name='Actual')
    y_pred = pd.Series(labels_pred, name='Predicted')
    df = pd.crosstab(y_actu, y_pred)
    for col in df.columns:
        act_label = 'p%d' % df[col].argmax()
        if act_label in df.columns:
            df[act_label] += df[col]
        else:
            df[act_label] = df[col]
        del df[col]

    df.sort_index(axis=1, inplace=True)
    tmp = df.values
    # wrong_nums = np.sum(tmp - np.diag(np.diag(tmp)), axis=1)
    # confusions = np.sum((tmp - np.diag(np.diag(tmp))).flatten())
    # confusions = confusions * 100. / len(labels_true)
    # df_perc = df.apply(lambda x: x / float(x.sum()), axis=1)
    # df_perc.applymap(lambda x: '%.2f%%' % x)
    # df = df_perc
    # return df, confusions, wrong_nums
    return sum(np.diag(tmp)) * 100. / len(labels_true)
    # return df


# labels_true=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 1, 0, 2, 2, 2, 0]
# labels_pred=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
# purity =calculate_purity(labels_true, labels_pred)
# print purity

def calculate_mean_euclidean_distances(centers_true, centers_estimated):
    """
    :param X: input data
    :param labels_true:
    :param centers_estimated: must be array
    :return:
    """
    # centers_true = [np.mean(X[labels_true == label], axis=0) for label in np.unique(labels_true)]
    result=[]
    # for center_true in centers_true:
    #     tmp=[]
    #     for center_estimated in centers_estimated:
    #         tmp.append(np.linalg.norm(center_true-center_estimated))
    #     result.append(min(tmp))
    for center_estimated in centers_estimated:
        tmp=[]
        for center_true in centers_true:
            tmp.append(np.linalg.norm(center_true-center_estimated))
        result.append(min(tmp))
    return np.mean(result)
    # return np.mean([min(map(np.linalg.norm,centers_estimated - center_true )) for center_true in centers_true])
