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
        # -- 1d scaled flatten thermal data --
        sscaler = preprocessing.StandardScaler()
        if scale_type == "individual":
            scaled_temperature = {
                k: sscaler.fit_transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scale individualy
            }
        elif scale_type == "all":
            sscaler.fit(all_temperature)
            scaled_temperature = {
                k: sscaler.transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scaled by all temperature
            }
        else:
            print("not supported scale type:",scale_type)
        self.scaled_temperature = scaled_temperature
        self.scaled_all_temperature = np.concatenate([*scaled_temperature.values()])
        # -- 1d masked flatter thermal data --
        masked_temperature = {
            k: temperature[k][(v>-2) & (v<2)].reshape(-1,3) for k, v in scaled_temperature.items()
        }
        self.masked_temperature = masked_temperature
        self.masked_all_temperature = np.concatenate([*masked_temperature.values()])        
        # -- 1d scaled masked flatter thermal data --
        scaled_masked_temperature = {
            k: sscaler.fit_transform(v) for k, v in masked_temperature.items()
        }
        self.scaled_masked_temperature = scaled_masked_temperature
        self.scaled_masked_all_temperature = np.concatenate([*scaled_masked_temperature.values()])
        # -- 1d transformed flatten thermal data --
        pscaler = preprocessing.PowerTransformer(standardize=True)
        if scale_type == "individual":
            transformed_temperature = {
                k: pscaler.fit_transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scale individualy
            }
        elif scale_type == "all":
            pscaler.fit(all_temperature)
            transformed_temperature = {
                k: sscaler.transform(v.reshape(-1,3)) for k, v in thermal_img_files.items() # scaled by all temperature
            }
        else:
            print("not supported scale type:",scale_type)
        self.transformed_temperature = transformed_temperature
        self.transformed_all_temperature = np.concatenate([*transformed_temperature.values()])
        

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
