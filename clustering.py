
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from xmeans import XMeans
from star_clustering import StarCluster

class TemperatureClusters:
    def __init__(self, data, n_clusters=10, method="kmeans"):
        self.method = method
        if method == "xmeans":
            self.model = self.get_xmeans_model(data)
            self.n_clusters = max(self.model.labels_)+1
        elif method == "dbscan":
            self.model = self.get_dbscan_model(data)
            self.n_clusters = max(self.model.labels_)+1
        elif method == "star":
            self.model = self.get_star_model(data)
            self.n_clusters = max(self.model.labels_)+1            
        else:
            self.n_clusters = n_clusters
            self.model = self.get_kmeans_model(data)
        self.labels = self.model.labels_
        
    def get_kmeans_model(self, data):
        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=0
        ).fit(data)
        return model

    def get_xmeans_model(self, data):
        model = XMeans(
        ).fit(data)
        return model

    def get_dbscan_model(self, data):
        model = DBSCAN(
            eps=0.27,
            min_samples=3,
        ).fit(data)
        return model

    def get_star_model(self, data):
        model = StarCluster(
        ).fit(data)
        return model

    def get_clusters_data(self, data):
        clusters = []
        assert data.shape[0] == self.labels.shape[0]
        for n in range(self.n_clusters):
            cluster = np.array([d for l, d in zip(self.labels,data) if l==n])
            clusters.append(cluster)
        return clusters
        #return [data[self.labels==n] for n in range(self.n_clusters)]
