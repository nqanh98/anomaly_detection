import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.stats import trim_mean

import utils

def get_max_num_hot_pixel_in_long_axis(hot_pixels):
    long_axis = np.argmax(hot_pixels.shape)
    return max(np.sum(hot_pixels,axis=long_axis))

def get_junction_box_fields(hot_pixels):
    long_axis, short_axis = np.argmax(hot_pixels.shape[:-1]), np.argmin(hot_pixels.shape[:-1])
    long_offset = int(hot_pixels.shape[long_axis] * 0.2)
    short_offset = int(hot_pixels.shape[short_axis] * 0.3)
    #print(long_offset, short_offset)
    edge1, edge2 = int(hot_pixels.shape[short_axis]/2 - short_offset), int(hot_pixels.shape[short_axis]/2 + short_offset)
    junction_box_fields = np.zeros(hot_pixels.shape)
    if long_axis == 0:
        junction_box_fields[ :long_offset, edge1:edge2, :] = 1
        junction_box_fields[-long_offset:, edge1:edge2, :] = 1
    elif long_axis == 1:
        junction_box_fields[edge1:edge2,  :long_offset, :] = 1
        junction_box_fields[edge1:edge2, -long_offset:, :] = 1
    return junction_box_fields

def get_flag_cluster_anomaly(hot_pixels):
    n_hot_pixel_in_long_axis = get_max_num_hot_pixel_in_long_axis(hot_pixels)
    #if n_hot_pixel_in_long_axis == max(hot_pixels.shape):
    offset = 3
    if n_hot_pixel_in_long_axis > max(hot_pixels.shape) - offset:
        return True
    else:
        return False

def get_flag_junction_box_error(hot_pixels):
    junction_box_fields = get_junction_box_fields(hot_pixels)
    junction_box_pixels = hot_pixels * junction_box_fields
    count_diff = np.sum(hot_pixels != junction_box_pixels)
    #if (hot_pixels == hot_pixels * junction_box_fields).all():
    print( np.sum(junction_box_pixels), count_diff)
    if np.sum(junction_box_pixels) > 4 and count_diff < 12:
        return True
    else:
        return False
        
def get_hot_counts(hot_pixels, clusters):
    cluster_2dmap = clusters.labels.reshape(hot_pixels.shape) + 1
    cluster_2dmap = cluster_2dmap * hot_pixels
    pos_hot_clusters = np.stack(np.where(cluster_2dmap > 0), axis=1)
    if len(pos_hot_clusters) > 1:
        Z = linkage(pdist(pos_hot_clusters), 'single')
        merged_hot_clusters = fcluster(Z, 1.0, criterion='distance')
        hot_counts = max(merged_hot_clusters)
    else:
        hot_counts = 0
    return hot_counts
    
def detect_module_type(hot_pixels, clusters):
    hot_counts = get_hot_counts(hot_pixels, clusters)
    flag_cluster_anomaly = get_flag_cluster_anomaly(hot_pixels)
    flag_junction_box_error = get_flag_junction_box_error(hot_pixels)
    if hot_pixels.mean() >= 0.5:
        module_type = "Module-Anomaly"        
    elif flag_cluster_anomaly and hot_pixels.mean() >= 0.25:
        module_type = "Cluster-Anomaly"
    elif flag_junction_box_error and hot_counts > 0:
        module_type = "Junction-Box-Error"
    elif hot_counts >= 2:
        module_type = "Multi-Hotspots"        
    elif hot_counts == 1:
        module_type = "Single-Hotspot"
    else:
        module_type = "Normal"
    return module_type    

def show_modules(img_dict, vmin=0, vmax=255):
    fig = plt.figure(figsize=(12,4),facecolor="w")
    n = len(img_dict)
    ax = {}
    for i, (k, v) in enumerate(img_dict.items()):
        ax[i] = fig.add_subplot(1,n,i+1)
        ax[i].imshow(v, vmin=vmin, vmax=vmax)
        ax[i].set_title(k)
    plt.show()

def remove_useless_clusters(hot_pixels):
    gray = hot_pixels.astype(np.uint8)
    #contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)        
        box = np.int0(cv2.boxPoints(rect))        
        peri = cv2.arcLength(cnt, True)
        peri_cnv = cv2.arcLength(cv2.convexHull(cnt), True)
        circularity = 4 * np.pi * area / peri**2 if peri > 0 else 0 
        waveness_shape_factor = peri_cnv / peri if peri > 0 else 0
        print(area, peri, circularity, waveness_shape_factor)        
        if area < 4.0: # remove small clusters
            #cv2.drawContours(gray, cnt, -1, color=(0,0,0), thickness=-1)            
            cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=1)
            cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=-1)
        if circularity < 0.25: # remove except circles and squares
            #cv2.drawContours(gray, cnt, -1, color=(0,0,0), thickness=-1)                        
            cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=1)
            cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=-1)
        if waveness_shape_factor < 0.7: # remove non-convex clusters
            #cv2.drawContours(gray, cnt, -1, color=(0,0,0), thickness=-1)            
            cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=-1)
            cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=1)            
    return gray

def get_hotspots_by_zscore(
        clusters_temperature, img_file, clusters, threshold=3.0, log=False):
    # -- hot cluster --
    hot_clusters = (clusters_temperature.mean(axis=1) > threshold)
    # -- hot pixel --
    hot_pixels = np.array([1 if c in np.where(hot_clusters==True)[0] else 0 for c in clusters.labels]) 
    hot_pixels = hot_pixels.reshape(*img_file.shape[:2],1)    
    if log:
        print("Hot cluster labels in module:",hot_clusters)
        print("Hotspot weights in module:",hot_pixels.mean())
    return hot_pixels, hot_clusters

def get_hotspots_by_models(
        clusters_temperature, transformed_clusters_temperature,
        img_file, clusters, model, log=False):
    # -- hot cluster --
    hot_clusters = (model.predict(clusters_temperature) < 0) \
        & (transformed_clusters_temperature.mean(axis=1) > 0)
    print(model.score_samples(clusters_temperature))
    print(model.offset_)
    # -- hot pixel --
    hot_pixels = np.array([1 if c in np.where(hot_clusters==True)[0] else 0 for c in clusters.labels])    
    hot_pixels = hot_pixels.reshape(*img_file.shape[:2],1) 
    return hot_pixels, hot_clusters

class AnoModels():
    def __init__(self):
        self.gamma = 1.5
        self.zscaler = {}
        self.lof = {}; self.lof_gamma = {}
        self.isof = {}; self.isof_gamma = {}

    def get_offset(self,thermal_data,module_labels,show=False):
        x = np.array(
            #[ thermal_data[c].all_temperature.mean() for c in range(0,max(module_labels)+1) ]
            #[ np.median(thermal_data[c].all_temperature) for c in range(0,max(module_labels)+1) ]
            [ trim_mean(thermal_data[c].all_temperature,0.2) for c in range(0,max(module_labels)+1) ]
        ).reshape(-1,1)
        #mscaler = MinMaxScaler([-0.5,2.0])
        mscaler = MinMaxScaler([-0.2,1.0])
        #mscaler = MinMaxScaler([-0.5,3.0])
        z = mscaler.fit_transform(x)
        offset = z / max(z)
        #offset = np.zeros(len(x))
        if show:
            fig = plt.figure(facecolor="w")
            plt.scatter(x,offset)
            plt.show()
        return offset
        
    def fit(self,thermal_data,module_labels):
        self.offset = self.get_offset(thermal_data,module_labels,show=False)
        for c in tqdm(range(0,max(module_labels)+1)):
            # -- temperatures -- 
            #n_modules = 100
            n_modules = len(thermal_data[c].temperature)
            all_temperature = thermal_data[c].all_temperature
            clusters_temperature = thermal_data[c].clusters_temperature
            #min_threshold = np.quantile(clusters_temperature,0.1)
            #max_threshold = np.quantile(clusters_temperature,0.9)
            #indices = (clusters_temperature > min_threshold) & (clusters_temperature < max_threshold)
            # -- Zscaler --
            self.zscaler[c] = preprocessing.RobustScaler().fit(
                utils.gamma_correction(all_temperature, gamma=self.gamma)
            )
            # -- Local Outlier Factor --
            lof = LocalOutlierFactor(n_neighbors=n_modules, contamination="auto", novelty=True)
            self.lof[c] = lof.fit(clusters_temperature)
            #self.lof[c] = lof.fit(clusters_temperature[indices].reshape(-1,3))
            self.lof[c].offset_ = -1.6 - 0.5 * self.offset[c] # default: -1.5
            #lof_gamma = LocalOutlierFactor(n_neighbors=n_modules, contamination="auto", novelty=True)
            #self.lof_gamma[c] = lof_gamma.fit(
            #    utils.gamma_correction(clusters_temperature, gamma=self.gamma)
            #)
            #self.lof_gamma[c].offset_ = -1.5 - 0.5 * self.offset[c] # default: -1.5
            # -- Isolation Forest --
            isof = IsolationForest(contamination="auto")
            self.isof[c] = isof.fit(clusters_temperature)
            #self.isof[c] = isof.fit(clusters_temperature[indices].reshape(-1,3))
            self.isof[c].offset_ = -0.6 - 0.2 * self.offset[c] # default: -0.5
            #isof_gamma = IsolationForest(contamination="auto")            
            #self.isof_gamma[c] = isof_gamma.fit(
            #    utils.gamma_correction(clusters_temperature, gamma=self.gamma) 
            #)
            #self.isof_gamma[c].offset_ = -0.6 - 0.1 * self.offset[c] # default: -0.5

    def check_pred_labels(self,thermal_data,module_labels,model1,model2):
        cmap = plt.get_cmap("tab20")
        for c in range(0,max(module_labels)+1):
            print("array offset:", c, thermal_data[c].all_temperature.mean(), model1[c].offset_, model2[c].offset_)
            fig = plt.figure(facecolor="w", figsize=(12,4))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            X1 = thermal_data[c].clusters_temperature
            X2 = thermal_data[c].clusters_temperature
            pred1 = model1[c].predict(X1)
            pred2 = model2[c].predict(X2)
            ax1.set_title("model1")
            ax2.set_title("model2")
            ax1.scatter(X1[:, 0], X1[:, 1], color=cmap(pred1+1))
            ax2.scatter(X2[:, 0], X2[:, 1], color=cmap(pred2+1))
            plt.show()            
