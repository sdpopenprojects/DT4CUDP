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
