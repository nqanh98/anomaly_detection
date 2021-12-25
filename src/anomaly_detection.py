import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

class HotspotDetectors():
    def __init__(self):
        # -- LocalOutlierFactor --
        self.lof = {}
        self.offset_lof = -4.0
        # -- IsolationForest --
        self.isof = {}
        self.offset_isof = -0.75
        # -- RobustZscore --
        self.gamma = 3.0
        self.min_zscore = 4.0
        
    def fit(self, thermal_data, module_labels):
        for c in tqdm(range(0,max(module_labels)+1)):
            # -- temperatures -- 
            #n_modules = 100
            n_modules = len(thermal_data[c].temperature)
            all_temperature = thermal_data[c].all_temperature
            clusters_temperature = thermal_data[c].clusters_temperature
            # -- Local Outlier Factor --
            lof = LocalOutlierFactor(n_neighbors=n_modules, contamination="auto", novelty=True)
            self.lof[c] = lof.fit(clusters_temperature)
            self.lof[c].offset_ = self.offset_lof # default: -1.5
            # -- Isolation Forest --
            isof = IsolationForest(contamination="auto")
            self.isof[c] = isof.fit(clusters_temperature)
            self.isof[c].offset_ = self.offset_isof # default: -0.5

    def check_pred_labels(self, thermal_data, module_labels, detectors):
        cmap = plt.get_cmap("tab20")
        get_label = lambda x: 'Outlier' if x == -1 else 'Inlier'
        for c in range(0,max(module_labels)+1):
            print("array: {} / temperature: {} / offsets: {} {}".format(
                c, thermal_data[c].all_temperature.mean(),detectors.lof[c].offset_, detectors.isof[c].offset_))
            fig = plt.figure(facecolor="w", figsize=(12,4))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            X = thermal_data[c].clusters_temperature
            label1 = list(map(get_label, detectors.lof[c].predict(X)))
            label2 = list(map(get_label, detectors.isof[c].predict(X)))
            df1 = pd.DataFrame({"Temperature":X.mean(axis=1),"label":label1})
            df2 = pd.DataFrame({"Temperature":X.mean(axis=1),"label":label2})
            ax1.set_title("Local Outlier Factor")
            ax2.set_title("Isolation Forest")
            sns.boxplot(x=[""]*len(X), y="Temperature", data=df1, ax=ax1, palette="pastel")
            sns.swarmplot(x=[""]*len(X), y="Temperature", hue="label", hue_order=["Inlier", "Outlier"], data=df1, ax=ax1)
            sns.boxplot(x=[""]*len(X), y="Temperature", data=df2, ax=ax2, palette="pastel")
            sns.swarmplot(x=[""]*len(X), y="Temperature", hue="label", hue_order=["Inlier", "Outlier"], data=df2, ax=ax2)            
            plt.show()            

class AnomalyTypeClassifier():
    def __init__(self, detectors):
        self.detectors = detectors
        # -- hotspot shape factors --
        self.min_hotspot_size = 4
        self.min_circularity = 0.25
        self.min_waveness_shape_factor = 0.7
        # -- module anomaly --
        self.min_module_anomaly_ratio = 0.5
        # -- cluster anomaly --
        self.min_cluster_anomaly_ratio = 0.2
        self.cluster_anomaly_offset = 0.2
        # -- junction box error -
        self.junction_box_offset_long = 0.2
        self.junction_box_offset_short = 0.3
        self.junction_box_offset_count = 12
        # -- RobustZscore --
        self.gamma = detectors.gamma
        self.min_zscore = detectors.min_zscore

    def get_clusters_temperature(self, clusters, temperature):
        sliced_data = clusters.get_clusters_data(temperature)
        clusters_temperature = np.stack([t.mean(axis=0).astype(temperature.dtype) for t in sliced_data])
        return clusters_temperature

    def get_hotspots_by_zscore(
            self, clusters_temperature, img_file, clusters, threshold=3.0, log=False):
        # -- hot cluster --
        hot_clusters = (clusters_temperature.mean(axis=1) > threshold)
        print(clusters_temperature.mean(axis=1))
        print(threshold)
        # -- hot pixel --
        hot_pixels = np.array([1 if c in np.where(hot_clusters==True)[0] else 0 for c in clusters.labels]) 
        hot_pixels = hot_pixels.reshape(*img_file.shape[:2],1)    
        if log:
            print("Hot cluster labels in module:",hot_clusters)
            print("Hotspot weights in module:",hot_pixels.mean())
        return hot_pixels

    def get_hotspots_by_models(
            self, clusters_temperature, transformed_clusters_temperature,
            img_file, clusters, model, log=False):
        # -- hot cluster --
        hot_clusters = (model.predict(clusters_temperature) < 0) \
            & (transformed_clusters_temperature.mean(axis=1) > 0)
        print(model.score_samples(clusters_temperature))
        print(model.offset_)
        # -- hot pixel --
        hot_pixels = np.array([1 if c in np.where(hot_clusters==True)[0] else 0 for c in clusters.labels])    
        hot_pixels = hot_pixels.reshape(*img_file.shape[:2],1) 
        return hot_pixels

    def get_max_num_hot_pixel_in_long_axis(self, hot_pixels):
        long_axis = np.argmax(hot_pixels.shape)
        return max(np.sum(hot_pixels,axis=long_axis))

    def get_junction_box_fields(self, hot_pixels):
        long_axis, short_axis = np.argmax(hot_pixels.shape[:-1]), np.argmin(hot_pixels.shape[:-1])
        long_offset = int(hot_pixels.shape[long_axis] * self.junction_box_offset_long)
        short_offset = int(hot_pixels.shape[short_axis] * self.junction_box_offset_short)
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

    def get_flag_cluster_anomaly(self, hot_pixels):
        n_hot_pixel_in_long_axis = self.get_max_num_hot_pixel_in_long_axis(hot_pixels)
        #if n_hot_pixel_in_long_axis == max(hot_pixels.shape):
        offset = int(max(hot_pixels.shape) * self.cluster_anomaly_offset)
        if n_hot_pixel_in_long_axis > max(hot_pixels.shape) - offset:
            return True
        else:
            return False

    def get_flag_junction_box_error(self,hot_pixels):
        junction_box_fields = self.get_junction_box_fields(hot_pixels)
        junction_box_pixels = hot_pixels * junction_box_fields
        count_diff = np.sum(hot_pixels != junction_box_pixels)
        #print( np.sum(junction_box_pixels), count_diff)
        #if (hot_pixels == hot_pixels * junction_box_fields).all():    
        if np.sum(junction_box_pixels) > self.min_hotspot_size and count_diff < self.junction_box_offset_count:
            return True
        else:
            return False

    def get_hot_counts(self, hot_pixels, clusters):
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

    def remove_useless_clusters(self, hot_pixels):
        gray = hot_pixels.astype(np.uint8)
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
            if area < self.min_hotspot_size: # remove small clusters
                #cv2.drawContours(gray, cnt, -1, color=(0,0,0), thickness=-1)            
                cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=1)
                cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=-1)
            if circularity < self.min_circularity: # remove except circles and squares
                #cv2.drawContours(gray, cnt, -1, color=(0,0,0), thickness=-1)                        
                cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=1)
                cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=-1)
            if waveness_shape_factor < self.min_waveness_shape_factor: # remove non-convex clusters
                #cv2.drawContours(gray, cnt, -1, color=(0,0,0), thickness=-1)            
                cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=-1)
                cv2.drawContours(gray, [box], -1, color=(0,0,0), thickness=1)            
        return gray

    def get_module_type(self, hot_pixels, clusters):
        hot_counts = self.get_hot_counts(hot_pixels, clusters)
        flag_cluster_anomaly = self.get_flag_cluster_anomaly(hot_pixels)
        flag_junction_box_error = self.get_flag_junction_box_error(hot_pixels)
        hot_ratio = hot_pixels.mean()
        if hot_ratio >= self.min_module_anomaly_ratio:
            module_type = "Module-Anomaly"        
        elif flag_cluster_anomaly and hot_ratio >= self.min_cluster_anomaly_ratio:
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

    def display_hotspots(self, thermal_img_file, clusters, clusters_temperature, module_type,
                         hot_pixels, hot_pixels_module, hot_pixels_lof, hot_pixels_isof):
        img_clustered = clusters_temperature[clusters.labels].reshape(thermal_img_file.shape)
        img_hotspots = img_clustered * hot_pixels
        img_hotspots_module = img_clustered * hot_pixels_module
        img_hotspots_lof = img_clustered * hot_pixels_lof
        img_hotspots_isof = img_clustered * hot_pixels_isof
        if module_type not in ["Normal", "Junction-Box-Error"]:
            utils.show_img({
                "Original": thermal_img_file, 
                "Gamma": utils.gamma_correction(thermal_img_file, gamma=self.gamma), 
                "Hotspots": img_hotspots,
                "module": img_hotspots_module,
                "lof": img_hotspots_lof,
                "isof": img_hotspots_isof,
            })  

    def run(self, thermal_img_files, thermal_data, module_labels, 
            input_dir_path, list_target_modules = None):
        anomaly_modules = {}
        for n, k in enumerate(list(thermal_img_files)):    
            # -- module label --
            c = module_labels[n]
            if c == -1: # skip non-grouped modules
                pass
            elif list_target_modules == None or k in list_target_modules: # narrowing down if target list is given
                # -- clustering --
                thermal_img_file = thermal_img_files[k]
                clusters = thermal_data[c].clusters[k]
                # -- temperatures -- 
                temperature = thermal_data[c].temperature[k]
                gamma_temperature = utils.gamma_correction(temperature, gamma=self.gamma)
                scaled_temperature = preprocessing.RobustScaler().fit_transform(gamma_temperature)
                # -- clusters temperatures --
                clusters_temperature = self.get_clusters_temperature(clusters, temperature)
                scaled_clusters_temperature = self.get_clusters_temperature(clusters, scaled_temperature)
                # -- hot spot detection --    
                hot_pixels_module = self.get_hotspots_by_zscore(
                    scaled_clusters_temperature, thermal_img_file, clusters, threshold=self.min_zscore, log=False)   
                hot_pixels_lof = self.get_hotspots_by_models(
                    clusters_temperature, scaled_clusters_temperature,
                    thermal_img_file, clusters, self.detectors.lof[c], log=False)        
                hot_pixels_isof = self.get_hotspots_by_models(
                    clusters_temperature, scaled_clusters_temperature,
                    thermal_img_file, clusters, self.detectors.isof[c], log=False)
                # -- map to anomaly type  --
                hot_pixels = (hot_pixels_module | hot_pixels_lof | hot_pixels_isof) # take "or" results from hotspot detectors
                hot_pixels = self.remove_useless_clusters(self.remove_useless_clusters(hot_pixels)) # doubly preformed just in case
                module_type = self.get_module_type(hot_pixels, clusters)                          
                # -- save anomaly modules --
                print(k, module_type)
                if module_type not in anomaly_modules:
                    anomaly_modules[module_type] = [k]
                else:
                    anomaly_modules[module_type].append(k)
                # -- display -- 
                self.display_hotspots(thermal_img_file, clusters, clusters_temperature, module_type,
                                      hot_pixels, hot_pixels_module, hot_pixels_lof, hot_pixels_isof)
        return anomaly_modules
        
