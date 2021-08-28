
import numpy as np
from sklearn.cluster import KMeans

def get_kmeans(inputs, n_clusters=8):
    model = KMeans(n_clusters=n_clusters).fit(inputs)
    return model

def get_clusters_data(model, data):
    clusters = []
    assert data.shape[0] == model.labels_.shape[0]
    for n in range(model.n_clusters):
        cluster = np.array(
            [ d for l, d in zip(model.labels_,data ) if l==n ]
        )
        clusters.append(cluster)
    return clusters
