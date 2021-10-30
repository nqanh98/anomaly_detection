import cv2
import glob
import os

import numpy as np
from sklearn import preprocessing

def get_img_files(dir_path="./images/modules", gray=True):
    img_files = {}
    for filepath in glob.glob(dir_path + "/*.jpg"):
        filename = os.path.basename(filepath) 
        img = cv2.imread(filepath)
        if gray:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_files[filename] = img_gray
        else:
            img_files[filename] = img
    return img_files

class ThermalData:
    def __init__(self, thermal_img_files, scale_type="individual"):
        self.transforms(thermal_img_files, scale_type)
        self.transforms_with_index(thermal_img_files, scale_type)
        
    def transforms(self, thermal_img_files, scale_type):
        # -- 1d flatten thermal data --
        temperature = {
            k: v.reshape(-1,v.shape[2]) for k, v in thermal_img_files.items()
        }
        self.temperature = temperature
        self.all_temperature = np.concatenate([*temperature.values()])
        self.mean_temperature = self.all_temperature.mean(axis=0)
        self.median_temperature = np.mean(self.all_temperature,axis=0)
        # -- 1d transformed flatten thermal data --
        #transformed_temperature = {
        #    k: preprocessing.PowerTransformer().fit_transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scale individualy
        #}
        #self.transformed_temperature = transformed_temperature
        #self.transformed_all_temperature = np.concatenate([*transformed_temperature.values()])
        # -- 1d scaled flatten thermal data --
        if scale_type == "individual":
            scaled_temperature = {
                k: preprocessing.RobustScaler().fit_transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scale individualy
                #k: preprocessing.RobustScaler().fit_transform(v.reshape(-1,3)) for k, v in transformed_temperature.items() # scale individualy
            }
        elif scale_type == "all":
            rscaler = preprocessing.RobustScaler()
            rscaler.fit(all_temperature)
            scaled_temperature = {
                k: rscaler.transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scaled by all temperature
            }
        else:
            print("not supported scale type:",scale_type)
        self.scaled_temperature = scaled_temperature
        self.scaled_all_temperature = np.concatenate([*scaled_temperature.values()])
        
    def get_data_with_index(self, data):
        data_with_index = []
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                idx = y * data.shape[1] + x
                data_with_index.append([*data[y][x], y, x])            
        return np.array(data_with_index)
        
    def transforms_with_index(self, thermal_img_files, scale_type):
        # -- 1d flatten thermal data with index --
        temperature_with_index = {
            k: self.get_data_with_index(v) for k, v in thermal_img_files.items()
        }
        self.temperature_with_index = temperature_with_index
        all_temperature_with_index = np.concatenate([*temperature_with_index.values()])
        # -- 1d scaled flatten thermal data with index --
        if scale_type == "individual":
            scaled_temperature_with_index = {
                #k: rscaler.fit_transform(self.get_data_with_index(v)) for k, v in thermal_img_files.items()
                k: preprocessing.RobustScaler().fit_transform(self.get_data_with_index(v)) for k, v in thermal_img_files.items()
            }
        elif scale_type == "all":
            rscaler = preprocessing.RobustScaler()
            rscaler.fit(all_temperature_with_index)
            scaled_temperature_with_index = {
                k: rscaler.transform(self.get_data_with_index(v)) for k, v in thermal_img_files.items()
            }
        else:
            print("not supported scale type:",scale_type)
        self.scaled_temperature_with_index = scaled_temperature_with_index
