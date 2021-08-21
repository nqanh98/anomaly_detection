import cv2
import glob
import os

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
