import sys
sys.path.append("../lib")
import contours_extractor as extractor
import shutil
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans, DBSCAN



# -- module utilities --


def plot_module_map_with_labels(img, module_contours, module_labels):
    fig, ax = plt.subplots(facecolor="w", figsize=(25, 20))
    colors = list(matplotlib.colors.XKCD_COLORS.items())[:max(module_labels)+1]
    module_centers = np.array([c.mean(axis=0) for c in module_contours])
    for i in range(-1, max(module_labels)+1):
        data = module_centers[module_labels == i]
        #plt.scatter(data[:, 0], data[:, 1], color=colors[i][1], label=str(i))
        if len(data) > 0:
            plt.scatter(data[:, 0], data[:, 1], color=colors[i][1])
            plt.annotate(str(i), (np.mean(data[:, 0]), np.mean(
                data[:, 1])), color="black", fontsize=18)
    #ax.legend(loc='upper left')
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([img.shape[0], 0])
    plt.show()


def split_module_labels(module_contours, module_labels, desired_cluster_size=50):
    splitted_module_labels = module_labels
    module_centers = np.array([c.mean(axis=0) for c in module_contours])
    for i in range(max(splitted_module_labels)+1):
        # coordinates for target label
        data = module_centers[splitted_module_labels == i]
        if len(data) >= 2*desired_cluster_size:
            # clustering by K-means
            model = KMeans(
                n_clusters=int(len(data)/desired_cluster_size)
            ).fit(data)
            # new labels for target
            new_labels = np.where(model.labels_ > 0,
                                  model.labels_ + max(splitted_module_labels), i)
            # replace with new labels
            splitted_module_labels[np.where(
                splitted_module_labels == i)] = new_labels
    return splitted_module_labels
# --


class Filters():

    def __init__(self):
        # -- parameters for preprocessing --
        self.get_limit_data()
        self.flip_flag = False
        self.clipLimit = 2.0
        self.tileGridSize = 8
        self.blur_kernel_size = 10
        self.bilateral_d = 5
        self.bilateral_sigmaColor = 200
        self.bilateral_sigmaSpace = 50
        self.sharp_kernel_value = 10.0
        self.inflate_flag = True
        self.gamma = 1.5
        self.window_size_list = [91, 101, 111]
        self.C = -7
        self.ensemble_flag = True
        # -- parameters for selecting contours --
        self.area_min = 500
        self.area_max = 2500
        self.aspect_min = 0.45
        self.aspect_max = 0.65

    def get_limit_data(self):
        try:
            # ???????????????????????????????????????????????????????????????
            self.lower_lim_pix_val = np.load('params/lower_lim_pix_val.npy')
            self.upper_lim_pix_val = np.load('params/upper_lim_pix_val.npy')
            #self.lower_lim_pix_val = None
            #self.upper_lim_pix_val = None
        except:
            self.lower_lim_pix_val = None
            self.upper_lim_pix_val = None

    def check_limit_data(self, img):
        if self.upper_lim_pix_val == None or self.lower_lim_pix_val == None:
            mean_pix_val = np.mean(img)
            double_std_pix_val = 2 * np.std(img, ddof=1)
            max_pix_val = np.amax(img)
            min_pix_val = np.amin(img)
            self.upper_lim_pix_val = mean_pix_val + double_std_pix_val
            self.lower_lim_pix_val = mean_pix_val - double_std_pix_val

    def get_module_contours(self, img):
        self.check_limit_data(img)

        img_processed = extractor.preprocessing(img,
                                                self.upper_lim_pix_val, self.lower_lim_pix_val,
                                                flip_flag=self.flip_flag,
                                                clipLimit=self.clipLimit,
                                                tileGridSize=(
                                                    self.tileGridSize, self.tileGridSize),
                                                blur_kernel_size=(
                                                    self.blur_kernel_size, self.blur_kernel_size),
                                                bilateral_d=self.bilateral_d,
                                                bilateral_sigmaColor=self.bilateral_sigmaColor,
                                                bilateral_sigmaSpace=self.bilateral_sigmaSpace,
                                                sharp_kernel_value=self.sharp_kernel_value,
                                                inflate_flag=self.inflate_flag,
                                                gamma=self.gamma,
                                                window_size_list=self.window_size_list,
                                                C=self.C,
                                                ensemble_flag=self.ensemble_flag)

        modules_contours = extractor.select_contours_and_minAreaRect(img_processed,
                                                                     area_min=self.area_min,
                                                                     area_max=self.area_max,
                                                                     aspect_min=self.aspect_min,
                                                                     aspect_max=self.aspect_max)
        return modules_contours


class Modules():

    def __init__(self, module_contours):
        self.modules_contours = module_contours
        #self.anomaly_modules = {}
        # if anomaly_modules is not None:
        #    for k, v in anomaly_modules.items():
        #        self.anomaly_modules[k] = v

    def get_anomaly_contours(self, anomaly_modules):
        anomaly_contours = {}
        # for k, v in self.anomaly_modules.items():
        for k, v in anomaly_modules.items():
            module_index = list(map(lambda x: np.int(x.split(".")[0]), v))
            anomaly_contours[k] = np.array(self.modules_contours)[module_index]
        return anomaly_contours

    def get_img_contours(self, img, index=False):
        #img_con = cv2.drawContours(np.zeros_like(img), self.modules_contours, -1, 255, -1)
        img_con = cv2.fillPoly(np.zeros_like(img), self.modules_contours, 255)
        if index:
            return self.add_index(img_con)
        else:
            return img_con

    def get_img_target_contours(self, img, target_contours, index=False, color=(0, 255, 255)):
        img_filled = self.fill_target_panels(img, target_contours, color)
        if index:
            return self.add_index(img_filled)
        else:
            return img_filled

    def fill_target_panels(self, img, target_contours, color):
        if len(img.shape) < 3:
            img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
        else:
            img_colored = img
        cv2.drawContours(img_colored, target_contours, -1, color, 2)
        return img_colored

    def add_index(self, img):
        if len(img.shape) < 3:
            img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
        else:
            img_colored = img
        for i in range(len(self.modules_contours)):
            mu = cv2.moments(self.modules_contours[i])
            # if (mu["m00"] != 0):
            x, y = int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])
            cv2.putText(img_colored, str(
                  i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), thickness=1)
        return img_colored

    def get_scaled_centers(self, module_contours):
        centers = np.array([c.mean(axis=0) for c in module_contours])
        l = [max(cv2.minAreaRect(c)[1]) for c in module_contours]
        # coordinate in module-scaled space
        scaled_centers = centers / np.mean(l)
        return scaled_centers

    def get_dbscan_labels(self, module_contours, eps=1.5, show=False):
        scaled_centers = self.get_scaled_centers(module_contours)
        # eps: hyper parameter (1.5 module size)
        model = DBSCAN(eps=eps, min_samples=3).fit(scaled_centers)
        if show:
            cmap = plt.get_cmap("tab10")
            plt.scatter(
                scaled_centers[:, 0], -scaled_centers[:, 1], color=cmap(model.labels_+1))
            plt.show()
        return model.labels_

    def extract_modules(self, img, output_dir_path):
        mult = 1.0   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
        # mult = 1.1   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
        img_box = img.copy()
        #img_box = cv2.cvtColor(im_con.copy(), cv2.COLOR_GRAY2BGR)
        filePath = output_dir_path+"/modules/"
        os.makedirs(filePath, exist_ok=True)
        shutil.rmtree(filePath)
        os.makedirs(filePath, exist_ok=True)

        for i, cnt in enumerate(self.modules_contours):

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit

            W = rect[1][0]
            H = rect[1][1]

            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            rotated = False
            angle = rect[2]

            if angle < -45:
                angle += 90
                rotated = True

            center = (int((x1+x2)/2), int((y1+y2)/2))
            size = (int(mult*(x2-x1)), int(mult*(y2-y1)))
            # cv2.circle(img_box, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

            cropped = cv2.getRectSubPix(img_box, size, center)
            cropped = cv2.warpAffine(cropped, M, size)

            croppedW = W if not rotated else H
            croppedH = H if not rotated else W

            croppedRotated = cv2.getRectSubPix(cropped,
                                               (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))

            # plt.imshow(croppedRotated,cmap='gray')
            cv2.imwrite(filePath+str(i)+".jpg", croppedRotated)
            # plt.show()

            # cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit
            # plt.imshow(img_box,cmap='gray')
            # plt.show()


if __name__ == "__main__":

    # ?????????????????????
    input_img_path = '../ModuleExtraction/hokuto/thermal/DJI_0123_R.JPG'
    thermal_npdat_path = "../ModuleExtraction/hokuto/thermal"
    with open('anomaly_modules.json', 'r') as f:
        anomaly_modules = json.load(f)

    # ?????????????????????
    filters = Filters(thermal_npdat_path)
    img_org = cv2.imread(input_img_path, 0)
    img_filtered = filters.apply_all_filters(img_org)
    print("# done: apply filters")

    # ??????????????????????????????
    modules = Modules(img_filtered, anomaly_modules)
    print("# done: extract module contours")

    # ??????????????????????????????
    img_con_index = modules.get_img_contours(img_org, index=True)
    show_img({"extracted modules": img_con_index},
             cmap="gray", figsize=(30, 30))
    img_con = modules.get_img_contours(img_org, index=False)
    img_mask = cv2.bitwise_and(img_org, img_con)
    img_mask_index = modules.add_index(img_mask)
    show_img({"extracted modules (overlay)": img_mask_index},
             cmap="gray", figsize=(30, 30))
    print("# done: display modules")

    # ???????????????????????????
    anomaly_contours = modules.get_anomaly_contours()
    for k, v in anomaly_contours.items():
        img_target_index = modules.get_img_target_contours(
            img_con, v, index=True)
        show_img({"highlighted modules": img_target_index},
                 cmap="gray", figsize=(30, 30))
    print("# done: display highlighed module")

    # ???????????????????????????
    string_anomaly_labels = modules.get_dbscan_labels(
        anomaly_contours["Module-Anomaly"])
    anomaly_contours["String-Anomaly"] = anomaly_contours["Module-Anomaly"][string_anomaly_labels >= 0]
    img_string_index = modules.get_img_target_contours(
        img_con, anomaly_contours["String-Anomaly"], index=True)
    show_img({"string-anomaly modules": img_string_index},
             cmap="gray", figsize=(30, 30))
    print("# done: display string-anomaly module")
