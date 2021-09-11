
import numpy as np
from sklearn.cluster import KMeans
from xmeans import XMeans

class TemperatureClusters:
    def __init__(self, data, n_clusters=8, method="kmeans"):
        self.method = method
        if method == "xmeans":
            self.model = self.get_xmeans_model(data)
            self.n_clusters = len(self.model.cluster_sizes_)
        else:
            self.n_clusters = n_clusters
            self.model = self.get_kmeans_model(data)
        self.labels = self.model.labels_
        self.centers = self.model.cluster_centers_
        
    def get_kmeans_model(self, data):
        model = KMeans(
            n_clusters=self.n_clusters,
        ).fit(data)
        return model

    def get_xmeans_model(self, data):
        model = XMeans(
        ).fit(data)
        return model
    
    def get_clusters_data(self, data):
        clusters = []
        assert data.shape[0] == self.model.labels_.shape[0]
        for n in range(self.n_clusters):
            cluster = np.array(
                [ d for l, d in zip(self.model.labels_,data ) if l==n ]
            )
            clusters.append(cluster)
        return clusters

