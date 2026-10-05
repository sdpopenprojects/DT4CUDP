import copy

import numpy as np
from pyclustering.cluster.somsc import somsc
from sklearn.metrics import accuracy_score

from Cluster.GetCluster import GetCluster
from Cluster.get_clustering_model import get_clustering_model_Parameters
from utilities.SC import SC
from utilities.labelingCluster import labelCluster_v4_raw

Classfier_model = {
    "Kmeans", "Agglomerative", "Birch", "Kmedoids", "MiniBatchKmeans",
    "MeanShift", "AP", "GMM",
}
GetCluster_model = {
    "Bsas", "Cure", "Dbscan", "Mbsas", "Optics", "Rock", "Syncsom", "Bang",
    "KmeansPlus", "clarans", "EMA", "Fcm", "Gmeans", "Ttsas", "Xmeans",
}


def _cluster_predict(model_name, X, n, loop):
    if model_name in Classfier_model:
        clf = get_clustering_model_Parameters(model_name, n, loop).getCLF()
        return clf.fit_predict(X)
    if model_name in GetCluster_model:
        instance = GetCluster(model_name, X, n, loop).getCLF()
        instance.process()
        clusters = instance.get_clusters()
        predict_y = [0] * len(X)
        if clusters:
            for gc in range(len(clusters)):
                for j in clusters[gc]:
                    predict_y[j] = gc
        return predict_y
    if model_name == "Somsc":
        somsc_instance = somsc(X, n)
        somsc_instance.process()
        return somsc_instance.predict(X)
    if model_name == "SC":
        return SC(X)
    raise ValueError(f"Unknown model: {model_name}")


def G2PC(pre_label, test_cluster, test_orig, test_label, model_name, n, loop):
    """Permutation importance under the revised labelling protocol.

    Clustering uses the (possibly transformed) features. Cluster labels use
    the original metrics through labelCluster_v4_raw. Shuffling feature i
    applies the same permutation to that column in both matrices, then
    refits the clusterer.
    """
    drops = []
    baseline = accuracy_score(test_label, pre_label)
    test_cluster = np.asarray(test_cluster, dtype=float)
    test_orig = np.asarray(test_orig, dtype=float)
    for i in range(test_cluster.shape[1]):
        acc = []
        for _k in range(5):
            clustered = copy.deepcopy(test_cluster)
            original = copy.deepcopy(test_orig)
            order = np.random.permutation(len(test_cluster))
            clustered[:, i] = test_cluster[order, i]
            original[:, i] = test_orig[order, i]
            predict_y = _cluster_predict(model_name, clustered, n, loop)
            predict_y = labelCluster_v4_raw(original, predict_y)
            predict_y = np.asarray(predict_y).flatten()
            acc.append(baseline - accuracy_score(test_label, predict_y))
        drops.append(acc)
    return np.mean(np.array(drops), axis=1)
