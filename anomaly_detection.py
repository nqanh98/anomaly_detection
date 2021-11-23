import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
import itertools
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

temperature_classes = ["Very High", "High", "Medium", "Low", "Exception"]

class AnoGMM:
    def __init__(self, data):
        self.data = data
        self.model = self.get_gmm_model(data)
        self.index2class = self.get_index2class()
        
    def get_gmm_model(self, data):
        model = GaussianMixture(
            n_components=len(temperature_classes),
            covariance_type='spherical'
        ).fit(data)
        return model

    def get_index2class(self):
        index2class = {
            idx:c for idx, c
            in zip(np.argsort(self.model.means_[:, -1])[::-1],temperature_classes)
        }
        return index2class

    def display(self):
        fig = plt.figure(figsize=(12,4),facecolor="w")
        ax = fig.add_subplot(1,1,1)
        x = np.linspace(self.data.min(), self.data.max(), 300)        
        gd = []
        for idx, c in self.index2class.items():
            gd = stats.norm.pdf(x, self.model.means_[idx, -1], np.sqrt(self.model.covariances_[idx]))
            ax.plot(x, self.model.weights_[idx] * gd, label=c)
        ax.legend(loc='upper left')
        #ax.legend(loc='upper right')
        ax.set_xlabel('pxil values')
        ax.set_ylabel('prob')
        plt.show()

    def predict(self,data):
        return self.model.predict(data)

def get_max_num_hot_pixel_in_long_axis(hot_pixels):
    axis = np.argmax(hot_pixels.shape)
    return max(np.sum(hot_pixels,axis=axis))

def get_hot_counts(hot_clusters, hot_pixels, clusters):
    cluster_2dmap = clusters.labels.reshape(*hot_pixels.shape[:2])
    pos_hot_clusters = []
    for c in np.where(hot_clusters==True)[0]:
        pos_hot_clusters.extend(np.stack(np.where(cluster_2dmap == c), axis=1))
    if len(pos_hot_clusters) > 0:
        Z = linkage(pdist(pos_hot_clusters), 'single')
        merged_hot_clusters = fcluster(Z, 1.0, criterion='distance')
        hot_counts = max(merged_hot_clusters)
    else:
        hot_counts = sum(hot_clusters)
    return hot_counts
    
def detect_module_type(hot_clusters, hot_pixels, clusters):
    hot_counts = get_hot_counts(hot_clusters, hot_pixels, clusters)
    n_hot_pixel_in_long_axis = get_max_num_hot_pixel_in_long_axis(hot_pixels)
    if hot_pixels.mean() >= 0.8:
        module_type = "Module-Anomaly"        
    elif hot_pixels.mean() >= 1.0/4.0 and n_hot_pixel_in_long_axis == max(hot_pixels.shape):
        module_type = "Cluster-Anomaly"
    elif hot_counts >= 2:
        module_type = "Multi-Hotspots"        
    elif hot_counts == 1:
        module_type = "Single-Hotspot"
    else:
        module_type = "Normal"
    return module_type    

# clusters temperature 
# array_clusters_temperature = {}
# for c in range(0,max(module_labels)+1):
#     tmp = []
#     print("array:", c)
#     print("--> generate clusters temperature")
#     # -- clustering --
#     for k, v in data_array[c].scaled_temperature_with_index.items():
#         clusters = clustering.TemperatureClusters(v, method=clustering_method)  
#         sliced_data = clusters.get_clusters_data(data_array[c].temperature[k])   
#         original_clusters_temperature = np.stack([np.uint8(t.mean(axis=0)) for t in sliced_data])
#         tmp.append(original_clusters_temperature)
#     array_clusters_temperature[c] = np.vstack(tmp)

# # lof model
# array_clf = {}
# for c in range(0,max(module_labels)+1):
#     print("array:", c)
#     print("--> generate lof model")
#     n_modules = len(data_array[c].temperature)
#     lof = LocalOutlierFactor(n_neighbors=n_modules, contamination="auto", novelty=True)
#     #lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1, novelty=True)
#     array_clf[c] = lof.fit(array_clusters_temperature[c])

# cmap = plt.get_cmap("tab20")
# for c in range(0,max(module_labels)+1):
#     print("array:", c)
#     fig, ax = plt.subplots(facecolor="w")
#     data = array_clusters_temperature[c]
#     pred = array_clf[c].predict(data)
#     plt.scatter(data[:, 0], data[:, 1], color=cmap(pred+1))
#     ax.legend(loc='upper left')
#     plt.show()
