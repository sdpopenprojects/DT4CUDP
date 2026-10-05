import numpy as np
import pandas as pd
from numpy import int64
from sklearn import preprocessing

# labeling cluster according to the original metric values of modules
# fea: the original feature matrix
def labelCluster(fea, clus_label):
    fea = pd.DataFrame(fea)
    fea['clus_label'] = clus_label

    unique_labels = fea['clus_label'].unique()

    # computing overall mean value of each cluster
    cluster_means = {}
    for label in unique_labels:
        cluster_data = fea[fea['clus_label'] == label].iloc[:, :-1]
        cluster_means[label] = cluster_data.mean().mean()  

    # computing mean value of all clusters
    total_mean = sum(cluster_means.values()) / len(cluster_means)

    #  label cluster：smaller than total_mean -→ 0，large than total_mean -→ 1
    new_labels = []
    for label in clus_label:
        if cluster_means[label] <= total_mean:
            new_labels.append(0)
        else:
            new_labels.append(1)

    return new_labels


def _label_cluster_by_means(fea, clus_label, standardize=True):
    """Label the cluster with the higher metric mean as defective (1).

    standardize=True applies an extra per-feature z-score before the means
    are compared (labelCluster_v4). standardize=False uses the supplied
    metrics on their original scale (labelCluster_v4_raw).
    """
    fea = np.asarray(fea, dtype=float)
    if standardize:
        fea = preprocessing.scale(fea)
    fea = pd.DataFrame(fea)
    fea["clus_label"] = clus_label

    unique_labels = fea["clus_label"].unique()
    cluster_means = {}
    for label in unique_labels:
        cluster_data = fea[fea["clus_label"] == label].iloc[:, :-1]
        cluster_means[label] = cluster_data.mean().mean()

    total_mean = sum(cluster_means.values()) / len(cluster_means)
    new_labels = []
    for label in clus_label:
        if cluster_means[label] <= total_mean:
            new_labels.append(0)
        else:
            new_labels.append(1)
    return new_labels


def labelCluster_v4(fea, clus_label):
    return _label_cluster_by_means(fea, clus_label, standardize=True)


def labelCluster_v4_raw(fea, clus_label):
    return _label_cluster_by_means(fea, clus_label, standardize=False)
