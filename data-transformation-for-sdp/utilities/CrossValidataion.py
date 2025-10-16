import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# Different cross validations techniques: holdout, k-fold, stratified k-fold, bootstrap
# Please see https://scikit-learn.org/stable/api/sklearn.model_selection.html
# train_test_split, LeaveOneOut, KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold

def holdout(data, test_size, rand_seed=42):
    if not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y[y > 1] = 1

    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=test_size, random_state=rand_seed)

    return train_data, train_label, test_data, test_label


# sampling with replacement, whichever is not used in training data will be used in test data
# 0.632 bootstrap
def out_of_sample_bootstrap(data, rand_seed=42):
    if not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)

    indexs = list(data.index)
    train_idx = resample(indexs, n_samples=len(indexs), random_state=rand_seed)

    # picking rest of the data not considered in training data
    test_idx = list(set(indexs) - set(train_idx))

    train_data = data.iloc[train_idx, :-1]
    train_label = data.iloc[train_idx, -1]

    test_data = data.iloc[test_idx, :-1]
    test_label = data.iloc[test_idx, -1]

    train_label[train_label > 1] = 1
    test_label[test_label > 1] = 1

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label, train_idx, test_idx


# The ordinary bootstrap was proposed by Efron et al.
# The optimism_reduced bootstrap is an enhancement to the ordinary bootstrap that is used to correct for upward bias
# test twice: once using the original sample and
# again using the bootstrap sample from which the model was trained
def ordinary_bootstrap(data, rand_seed=42):
    if not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)

    indexs = list(data.index)
    train_idx = resample(indexs, n_samples=len(indexs), random_state=rand_seed)

    train_data = data.iloc[train_idx, :-1]
    train_label = data.iloc[train_idx, -1]

    # test data is the original samples
    test_data = data.iloc[:, :-1]
    test_label = data.iloc[:, -1]

    train_label[train_label > 1] = 1
    test_label[test_label > 1] = 1

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label

