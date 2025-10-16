from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, MiniBatchKMeans, \
    MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids


class get_clustering_model_Parameters(object):

    def __init__(self, model, n=2, random_state=None):
        self.model = model
        self.random_state = random_state
        self.n = n

    def getCLF(self):
        if self.model == 'Kmeans':
            model = KMeans(n_clusters=self.n, random_state=self.random_state)

        if self.model == 'GMM':
            model = GaussianMixture(n_components=self.n, random_state=self.random_state)

        if self.model == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=self.n)

        if self.model == 'Kmedoids':
            model = KMedoids(n_clusters=self.n, random_state=self.random_state)

        if self.model == 'Birch':
            model = Birch(n_clusters=self.n)

        if self.model == 'MiniBatchKmeans':
            model = MiniBatchKMeans(n_clusters=self.n, random_state=self.random_state)

        if self.model == 'MeanShift':
            model = MeanShift()

        if self.model == 'AP':
            model = AffinityPropagation(random_state=self.random_state)

        return model
