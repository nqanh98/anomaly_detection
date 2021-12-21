import re
import cv2
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing

from clustering import TemperatureClusters

# -- data utilities --
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_img_files(dir_path="./images/modules", gray=False):
    img_files = {}
    #for filepath in glob.glob(dir_path + "/*.jpg"):
    for filepath in sorted(glob.glob(dir_path + "/*.jpg"), key=natural_keys):
        filename = os.path.basename(filepath) 
        img = cv2.imread(filepath)
        if gray:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_files[filename] = img_gray
        else:
            img_files[filename] = img
    return img_files

def get_thermal_data(thermal_img_files, module_labels):
    data = {}; 
    for c in tqdm(range(0,max(module_labels)+1)):
        # get thermal data for target group
        indices = np.where(module_labels==c)
        target_thermal_img_files = pd.Series(thermal_img_files).iloc[indices]
        data[c] = ThermalData(target_thermal_img_files)
    return data
# --

class ThermalData:
    def __init__(self, thermal_img_files):
        self.transforms(thermal_img_files)
        self.transforms_with_index(thermal_img_files)
        self.clustering_with_temperature_and_index(thermal_img_files)

    def clustering_with_temperature_and_index(self, thermal_img_files):
        self.clusters = {}; tmp = []
        for k, v in self.scaled_temperature_with_index.items():
            self.clusters[k] = TemperatureClusters(v, method="kmeans")  
            sliced_data = self.clusters[k].get_clusters_data(self.temperature[k])   
            clusters_temperature = np.stack([np.uint8(t.mean(axis=0)) for t in sliced_data])
            tmp.append(clusters_temperature)
        self.clusters_temperature = np.vstack(tmp)            
        
    def transforms(self, thermal_img_files):
        # -- 1d flatten thermal data --
        temperature = {
            k: v.reshape(-1,v.shape[2]) for k, v in thermal_img_files.items()
        }
        self.temperature = temperature
        self.all_temperature = np.concatenate([*temperature.values()])
        # -- 1d scaled flatten thermal data --
        scaled_temperature = {
            k: preprocessing.RobustScaler().fit_transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scale individualy
        }
        self.scaled_temperature = scaled_temperature
        self.scaled_all_temperature = np.concatenate([*scaled_temperature.values()])
        
    def get_data_with_index(self, data):
        data_with_index = []
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                idx = y * data.shape[1] + x
                data_with_index.append([*data[y][x], y, x])
        return np.array(data_with_index)
        
    def transforms_with_index(self, thermal_img_files):
        # -- 1d flatten thermal data with index --
        temperature_with_index = {
            k: self.get_data_with_index(v) for k, v in thermal_img_files.items()
        }
        self.temperature_with_index = temperature_with_index
        # -- 1d scaled flatten thermal data with index --
        scaled_temperature_with_index = {
            k: preprocessing.RobustScaler().fit_transform(self.get_data_with_index(v)) for k, v in thermal_img_files.items()
        }
        self.scaled_temperature_with_index = scaled_temperature_with_index
