from pyclustering.cluster.bang import bang
from pyclustering.cluster.bsas import bsas
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.cure import cure
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.ema import ema
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.mbsas import mbsas
from pyclustering.cluster.optics import optics
from pyclustering.cluster.rock import rock
from pyclustering.cluster.syncsom import syncsom
from pyclustering.cluster.ttsas import ttsas
from pyclustering.cluster.xmeans import xmeans


class GetCluster(object):

    def __init__(self, classifier, test_data, n=2, random_state=None):
        self.clf = classifier
        self.test_data = test_data
        self.random_state = random_state
        self.n = n

    def getCLF(self):
        if self.clf == 'Bsas':
            instance = bsas(self.test_data, self.n, 1.0)

        if self.clf == 'Cure':
            instance = cure(self.test_data, self.n)

        if self.clf == 'Dbscan':
            instance = dbscan(self.test_data, 0.5, 3)

        if self.clf == 'Mbsas':
            instance = mbsas(self.test_data, self.n, 1.0)

        if self.clf == 'Optics':
            instance = optics(self.test_data, 0.5, 3)

        if self.clf == 'Rock':
            instance = rock(self.test_data, 1.0, 7)

        if self.clf == 'Syncsom':
            instance = syncsom(self.test_data, 4, 4, 1.0)

        if self.clf == 'Bang':
            instance = bang(self.test_data, 3)

        if self.clf == 'KmeansPlus':
            centers = kmeans_plusplus_initializer(self.test_data, 2,
                                                   kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
            instance = kmeans(self.test_data, centers)

        if self.clf == 'clarans':
            instance = clarans(self.test_data, 2, 10, 5)

        if self.clf == 'EMA':
            instance = ema(self.test_data, 3)

        if self.clf == 'Fcm':
            instance = kmeans_plusplus_initializer(self.test_data, 2,
                                                  kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
            instance = fcm(self.test_data, instance)

        if self.clf == 'Gmeans':
            instance = gmeans(self.test_data, repeat=10)

        if self.clf == 'Ttsas':
            instance = ttsas(self.test_data, 1.0, 2.0)

        if self.clf == 'Xmeans':
            instance = kmeans_plusplus_initializer(self.test_data, 2).initialize()
            instance = xmeans(self.test_data, instance, 20)

        return instance