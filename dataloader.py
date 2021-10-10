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
        all_temperature = np.concatenate([*temperature.values()])
        self.temperature = temperature
        self.all_temperature = all_temperature        
        # -- 1d scaled flatten thermal data --
        sscaler = preprocessing.StandardScaler()
        if scale_type == "individual":
            scaled_temperature = {
                k: sscaler.fit_transform(v.reshape(-1,v.shape[2])) for k, v in thermal_img_files.items() # scale individualy
            }
        elif scale_type == "all":
            sscaler.fit(all_temperature)
            scaled_temperature = {
                k: sscaler.transform(v.reshape(-1,v.shape[2])) for k, v in thermal_img_files.items() # scaled by all temperature
            }
        else:
            print("not supported scale type:",scale_type)
        scaled_all_temperature = np.concatenate([*scaled_temperature.values()])            
        self.scaled_temperature = scaled_temperature
        self.scaled_all_temperature = scaled_all_temperature

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
        sscaler = preprocessing.StandardScaler()
        if scale_type =="individual":
            scaled_temperature_with_index = {
                k: sscaler.fit_transform(self.get_data_with_index(v)) for k, v in thermal_img_files.items()
            }
        elif scale_type == "all":
            sscaler.fit(all_temperature_with_index)
            scaled_temperature_with_index = {
                k: sscaler.transform(self.get_data_with_index(v)) for k, v in thermal_img_files.items()
            }
        else:
            print("not supported scale type:",scale_type)
            
        self.scaled_temperature_with_index = scaled_temperature_with_index
