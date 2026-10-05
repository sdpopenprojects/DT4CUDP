"""Shared helpers for the revised clustering protocol.

Clusters are fit on the transformed out-of-bag test features.
Cluster labels are assigned from the original metrics with
labelCluster_v4_raw (no extra z-score inside the labeller).
"""
import os

import numpy as np
import pandas as pd
from pyclustering.cluster.somsc import somsc
from scipy.io import arff
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from Cluster.GetCluster import GetCluster
from Cluster.get_clustering_model import get_clustering_model_Parameters
from utilities.SC import SC
from utilities.column_deletion import remove_constant_columns
from utilities.labelingCluster import labelCluster_v4_raw
from utilities.quantile_rank import rank_transform_dataframe

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")

ARFF_PROJECTS = {"EQ", "JDT", "ML", "PDE", "LC"}
PROMISE_PROJECTS = {
    "ant-1.7", "camel-1.4", "ivy-2.0", "jedit-4.0", "log4j-1.0", "poi-2.0",
    "tomcat", "velocity-1.6", "xalan-2.4", "xerces-1.3",
}
JIRA_PROJECTS = {
    "activemq-5.0.0", "derby-10.5.1.1", "groovy-1_6_BETA_1", "hbase-0.94.0",
    "hive-0.9.0", "jruby-1.1", "wicket-1.3.0-beta2",
}
ALL_PROJECTS = [
    "EQ", "JDT", "ML", "PDE", "LC",
    "ant-1.7", "camel-1.4", "ivy-2.0", "jedit-4.0", "log4j-1.0",
    "poi-2.0", "tomcat", "velocity-1.6", "xalan-2.4", "xerces-1.3",
    "activemq-5.0.0", "derby-10.5.1.1", "groovy-1_6_BETA_1", "hbase-0.94.0",
    "hive-0.9.0", "jruby-1.1", "wicket-1.3.0-beta2",
]
ALL_NORMS = ["O", "log", "Z-score", "Max-Min", "yeo-johnson", "rank-transformation"]
ALL_CLF = [
    "Kmeans", "Kmedoids", "Xmeans", "Fcm", "Gmeans", "MiniBatchKmeans",
    "KmeansPlus", "Birch", "Cure", "Rock", "Agglomerative", "Dbscan",
    "Optics", "MeanShift", "Somsc", "Syncsom", "EMA", "GMM", "AP", "SC",
    "Bsas", "Mbsas", "Ttsas",
]
INTERPRET_CLF = ["Kmeans", "Kmedoids", "Gmeans", "AP", "SC"]
N_CLUSTERS = 2

_SKLEARN_MODELS = {
    "Kmeans", "Agglomerative", "Birch", "Kmedoids", "MiniBatchKmeans",
    "MeanShift", "AP", "GMM",
}
_PYCLUSTERING_MODELS = {
    "Bsas", "Cure", "Dbscan", "Mbsas", "Optics", "Rock", "Syncsom", "Bang",
    "KmeansPlus", "clarans", "EMA", "Fcm", "Gmeans", "Ttsas", "Xmeans",
}


def dataset_path(project_name, data_dir=DATA_DIR):
    if project_name in ARFF_PROJECTS:
        return os.path.join(data_dir, project_name + ".arff")
    return os.path.join(data_dir, project_name + ".csv")


def missing_datasets(names=None, data_dir=DATA_DIR):
    names = ALL_PROJECTS if names is None else names
    return [dataset_path(name, data_dir) for name in names if not os.path.isfile(dataset_path(name, data_dir))]


def load_project(project_name, data_dir=DATA_DIR):
    if project_name in ARFF_PROJECTS:
        data, _meta = arff.loadarff(dataset_path(project_name, data_dir))
        data = pd.DataFrame(data)
        data.iloc[:, -1] = data.iloc[:, -1].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
        data["new_col"] = data.iloc[:, -1].replace({"buggy": 1, "clean": 0}).astype(np.float64)
        data = data.drop(data.columns[-2], axis=1)
        bugs = data.iloc[:, -1]
        locs = data["ck_oo_numberOfLinesOfCode"]
    elif project_name in JIRA_PROJECTS:
        data = pd.read_csv(dataset_path(project_name, data_dir))
        data.iloc[:, -1] = (data.iloc[:, -1] != 0).astype(int)
        data = data.drop(data.columns[[0, -2, -3, -4]], axis=1)
        data = data.apply(pd.to_numeric, errors="coerce")
        bugs = data.iloc[:, -1]
        locs = data["CountLine"]
    elif project_name in PROMISE_PROJECTS:
        data = pd.read_csv(dataset_path(project_name, data_dir))
        data.iloc[:, -1] = (data.iloc[:, -1] != 0).astype(int)
        data = data.drop(data.columns[[0, 1, 2]], axis=1)
        data = data.apply(pd.to_numeric, errors="coerce")
        bugs = data.iloc[:, -1]
        locs = data["loc"]
    else:
        raise ValueError(f"Unknown project: {project_name}")
    return data, bugs, locs


def apply_transform(normalization_model, train_data, test_data):
    if normalization_model == "log":
        train_data = np.log(train_data + 1)
        test_data = np.log(test_data + 1)
        train_data[np.isneginf(train_data)] = 0
        test_data[np.isneginf(test_data)] = 0
        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
    elif normalization_model == "Z-score":
        train_data = preprocessing.scale(train_data)
        test_data = preprocessing.scale(test_data)
    elif normalization_model == "Max-Min":
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.fit_transform(test_data)
    elif normalization_model == "box-cox":
        test_data = remove_constant_columns(test_data)
        test_data = test_data + 1
        pt = PowerTransformer(method="box-cox")
        test_data = pt.fit_transform(test_data)
    elif normalization_model == "yeo-johnson":
        pt = PowerTransformer(method="yeo-johnson")
        test_data = pt.fit_transform(test_data)
    elif normalization_model == "rank-transformation":
        test_data = np.array(rank_transform_dataframe(test_data))
    elif normalization_model == "O":
        train_data = np.array(train_data)
        test_data = np.array(test_data)
    else:
        raise ValueError(f"Unknown normalization: {normalization_model}")
    return train_data, np.asarray(test_data, dtype=float)


def run_cluster(model_name, test_data, n, loop):
    if model_name in _SKLEARN_MODELS:
        clf = get_clustering_model_Parameters(model_name, n, loop).getCLF()
        return clf.fit_predict(test_data)
    if model_name in _PYCLUSTERING_MODELS:
        instance = GetCluster(model_name, test_data, n, loop).getCLF()
        instance.process()
        clusters = instance.get_clusters()
        predict_y = [0] * len(test_data)
        if clusters:
            for gc in range(len(clusters)):
                for j in clusters[gc]:
                    predict_y[j] = gc
        return predict_y
    if model_name == "Somsc":
        somsc_instance = somsc(test_data, n)
        somsc_instance.process()
        return somsc_instance.predict(test_data)
    if model_name == "SC":
        return SC(test_data)
    raise ValueError(f"Unknown model: {model_name}")


def label_predictions(test_orig, predict_y):
    predict_y = labelCluster_v4_raw(test_orig, predict_y)
    predict_y = np.asarray(predict_y).flatten()
    predict_y[predict_y > 1] = 1
    return predict_y


def count_finished_rounds(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def parse_names(text, allowed):
    if not text:
        return list(allowed)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise SystemExit(
            "Unknown name(s): " + ", ".join(unknown)
            + "\nAllowed: " + ", ".join(allowed)
        )
    return names
