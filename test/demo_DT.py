import os
import time
import warnings

import numpy as np
import pandas as pd
from pyclustering.cluster.somsc import somsc
from scipy.io import arff
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from Cluster.GetCluster import GetCluster
from Cluster.get_clustering_model import get_clustering_model_Parameters
from utilities.CrossValidataion import out_of_sample_bootstrap
from utilities.File import create_dir, save_results
from utilities.PerformanceMeasure import get_measure
from utilities.RankMeasure import rank_measure
from utilities.SC import SC
from utilities.labelingCluster import labelCluster

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = r'../result/'

    models = ['Kmeans', 'Agglomerative', 'Birch', 'Kmedoids', 'MiniBatchKmeans', 'MeanShift', 'AP', 'Bsas', 'Cure',
           'Dbscan', 'Mbsas', 'Optics', 'Rock', 'Somsc', 'Syncsom', 'KmeansPlus', 'EMA', 'Fcm', 'Gmeans',
           'Ttsas', 'Xmeans', 'GMM', 'SC']

    Classfier_model = {'Kmeans', 'Agglomerative', 'Birch', 'Kmedoids', 'MiniBatchKmeans', 'MeanShift', 'AP', 'GMM'}
    Special_model = {'ManualUp', 'Somsc'}
    GetCluster_model = {'Bsas', 'Cure', 'Dbscan', 'Mbsas', 'Optics', 'Rock', 'Syncsom', 'Bang', 'KmeansPlus', 'clarans',
                        'EMA', 'Fcm', 'Gmeans', 'Ttsas', 'Xmeans'}

    path = os.path.abspath('../data/')

    project_names = ['EQ', 'JDT', 'ML', 'PDE', 'LC', 'ant-1.7', 'camel-1.4', 'ivy-2.0', 'jedit-4.0', 'log4j-1.0',
                     'poi-2.0', 'tomcat', 'velocity-1.6', 'xalan-2.4', 'xerces-1.3', 'activemq-5.0.0', 'derby-10.5.1.1',
                     'groovy-1_6_BETA_1', 'hbase-0.94.0', 'hive-0.9.0', 'jruby-1.1', 'wicket-1.3.0-beta2']

    arff_project = {'EQ', 'JDT', 'ML', 'PDE', 'LC'}
    promise_project = {'ant-1.7', 'camel-1.4', 'ivy-2.0', 'jedit-4.0', 'log4j-1.0', 'poi-2.0', 'tomcat', 'velocity-1.6',
                       'xalan-2.4', 'xerces-1.3'}
    JIRA_projiect = {'activemq-5.0.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_1', 'hbase-0.94.0', 'hive-0.9.0', 'jruby-1.1',
                     'wicket-1.3.0-beta2'}

    normalization = ['O', 'log', 'Z-score', 'Max-Min']

    n = 2

    pro_num = len(project_names)

    Rep = 100

    for model_name in models:
        for normalization_model in normalization:
            for i in range(0, pro_num):
                project_name = project_names[i]

                if project_name in arff_project:
                    file = os.path.join(path, project_name + '.arff')
                    data, meta = arff.loadarff(file)
                    data = pd.DataFrame(data)

                    data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                    data['new_col'] = data.iloc[:, -1].replace({'buggy': 1, 'clean': 0}).astype(np.float64)
                    data = data.drop(data.columns[-2], axis=1)
                    bugs = data.iloc[:, -1]
                    LOCs = data['ck_oo_numberOfLinesOfCode']

                elif project_name in JIRA_projiect:
                    file = os.path.join(path, project_name + '.csv')
                    data = pd.read_csv(file)
                    data.iloc[:, -1] = (data.iloc[:, -1] != 0).astype(int)

                    data = data.drop(data.columns[[0, -2, -3, -4]], axis=1)
                    data = data.apply(pd.to_numeric, errors='coerce')
                    bugs = data.iloc[:, -1]
                    LOCs = data['CountLine']

                elif project_name in promise_project:
                    file = os.path.join(path, project_name + '.csv')
                    data = pd.read_csv(file)
                    data.iloc[:, -1] = (data.iloc[:, -1] != 0).astype(int)

                    data = data.drop(data.columns[[0, 1, 2]], axis=1)
                    data = data.apply(pd.to_numeric, errors='coerce')
                    bugs = data.iloc[:, -1]
                    LOCs = data['loc']

                for loop in range(0, Rep):
                    print(model_name + '-> ' + normalization_model + ' ' + project_name + ' ' + str(loop + 1) + '/' +
                          str(Rep) + ' round Start!')

                    train_data, train_label, test_data, test_label, train_idx, test_idx = out_of_sample_bootstrap(data, loop)
                    LOC = LOCs[test_idx]
                    bug = bugs[test_idx]

                    if normalization_model == 'log':
                        # log transformation
                        train_data = np.log(train_data + 1)
                        test_data = np.log(test_data + 1)
                        # replace -inf with 0
                        train_data[np.isneginf(train_data)] = 0
                        test_data[np.isneginf(test_data)] = 0
                        # replace NaN with 0
                        train_data = np.nan_to_num(train_data)
                        test_data = np.nan_to_num(test_data)
                    elif normalization_model == 'Z-score':
                        # z-score
                        train_data = preprocessing.scale(train_data)
                        test_data = preprocessing.scale(test_data)
                    elif normalization_model == 'Max-Min':
                        scaler = MinMaxScaler()
                        train_data = scaler.fit_transform(train_data)
                        test_data = scaler.fit_transform(test_data)
                    elif normalization_model == 'O':
                        train_data = np.array(train_data)
                        test_data = np.array(test_data)

                    # running time
                    start = time.perf_counter()

                    # model
                    if model_name in Classfier_model:
                        model_cluster = get_clustering_model_Parameters(model_name, n, loop).getCLF()
                        predict_y = model_cluster.fit_predict(test_data)
                    elif model_name in GetCluster_model:
                        instance = GetCluster(model_name, test_data, n, loop).getCLF()
                        instance.process()
                        clusters = instance.get_clusters()
                        predict_y = [0] * len(test_data)
                        if clusters:
                            for gc in range(len(clusters)):
                                for j in clusters[gc]:
                                    predict_y[j] = gc
                    elif model_name == 'ManualUp':
                        predict_y = [0] * len(test_data)
                    elif model_name == 'Somsc':
                        somsc_instance = somsc(test_data, n)
                        somsc_instance.process()
                        predict_y = somsc_instance.predict(test_data)
                    elif model_name == 'SC':
                        predict_y = SC(test_data)

                    predict_y = labelCluster(test_data, predict_y)

                    end = time.perf_counter()
                    t = end - start


                    if not isinstance(bug, np.ndarray):
                        bug = bug.to_numpy().flatten()
                    test_label = test_label.flatten()

                    predict_y = np.array(predict_y)
                    for y in range(len(predict_y)):
                        if predict_y[y] > 1:
                            predict_y[y] = 1
                    predict_y = predict_y.flatten()

                    precision, recall, pf, f_measure, AUC, g_measure, g_mean, bal, MCC = get_measure(test_label, predict_y)
                    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA = rank_measure(predict_y, LOC, test_label)

                    measure = [precision, recall, pf, f_measure, AUC, g_measure, g_mean, bal, MCC, Popt, Erecall,
                               Eprecision, Efmeasure, PMI, IFA, t]
                    fres = create_dir(save_path + model_name + '/' + normalization_model)
                    save_results(fres + project_name, measure)

    print('done!')