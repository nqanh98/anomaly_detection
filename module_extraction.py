
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def get_thermal_data(thermal_npdat_path):
    thermal_npdat_list = glob.glob(thermal_npdat_path + "/*.JPG")
    thermal_npdats = []
    for filename in thermal_npdat_list:
        temp = cv2.imread(filename, 0)
        thermal_npdats.append(temp)
    return thermal_npdats

def get_rect_info(rect):
    # https://teratail.com/questions/153373
    # 長方形の中心
    center = (rect[0] + rect[2]) / 2

    # 長さが異なる2つの辺の長さをそれぞれ計算する。
    vec1 = rect[0] - rect[1]
    vec2 = rect[1] - rect[2]
    vec1_len = np.linalg.norm(vec1)
    vec2_len = np.linalg.norm(vec2)

    # 長辺が幅、短辺が高さとする。
    if vec1_len < vec2_len:
        width, height = vec2_len, vec1_len
        vecw = vec2
    else:
        width, height = vec1_len, vec2_len
        vecw = vec1

        # x 軸と長辺のなす角を計算する。
    if np.isclose(vecw[0], 0):
        angle = 0
    else:
        angle = -np.arctan(vecw[1] / vecw[0])
        angle = np.rad2deg(angle)
      
    return {'center': center, 'angle': angle, 'width': width, 'height': height}

def get_limits(thermal_npdat_path):
    thermal_npdats = get_thermal_data(thermal_npdat_path)
    thermal_npdats_arr = np.asarray(thermal_npdats)
    # del thermal_npdats
    thermal_npdats_arr = thermal_npdats_arr.reshape((-1))
    correct_value = np.amin(thermal_npdats_arr)

    if correct_value < 0:
        # ガンマ補正のため（負の値では計算できない）
        thermal_npdats_arr = thermal_npdats_arr - correct_value
        # 画像も同じ補正をかける
        img_np_org = img_np_org - correct_value
    else:
        pass

    mean_pix_val = np.mean(thermal_npdats_arr)
    double_std_pix_val = 2 * np.std(thermal_npdats_arr, ddof=1)
    max_pix_val = np.amax(thermal_npdats_arr)
    min_pix_val = np.amin(thermal_npdats_arr)
    
    # これらの範囲を超えるものはこの値で置換する
    upper_lim_pix_val = mean_pix_val + double_std_pix_val
    lower_lim_pix_val = mean_pix_val - double_std_pix_val

    return lower_lim_pix_val, upper_lim_pix_val
    
def show_img(img_dict, cmap=None, figsize=(12,12)):
    fig = plt.figure(figsize=figsize, facecolor="w")
    n = len(img_dict)
    ax = {}
    for i, (k, v) in enumerate(img_dict.items()):
        ax[i] = fig.add_subplot(1,n,i+1)
        #ax[i].imshow(v, interpolation = 'nearest', cmap = 'gray')
        if cmap is not None:
            ax[i].imshow(v, cmap=cmap)
        else:
            ax[i].imshow(v)
        ax[i].set_title(k)
    fig.tight_layout()      
    fig.show()

def normalize(img, lower_lim_pix_val, upper_lim_pix_val, show=False):
    temp = np.where(img > upper_lim_pix_val, upper_lim_pix_val, img)
    img_normed = np.where(temp < lower_lim_pix_val, lower_lim_pix_val, temp)
    if show: show_img({"input":img, "normed": img_normed},cmap="gray")
    return img_normed

def clahe(img, show=False):
    img_8 = np.array(img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_8)
    if show: show_img({"input":img, "clahe": img_clahe},cmap="gray")
    return img_clahe

def blur(img, show=False):
    img_blured = cv2.blur(img, (3, 3))
    if show: show_img({"input":img, "blured": img_blured},cmap="gray")    
    return img_blured

def bilateral_filter(img, show=False):
    img_bilateral_filtered = cv2.bilateralFilter(img.astype(np.float32), 3, 200, 50)
    if show: show_img({"input":img, "filtered": img_bilateral_filtered},cmap="gray")    
    return img_bilateral_filtered

def sharpen(img, show=False):
    k = 1.0
    kernel = np.array([[-k, -k, -k], [-k, 1+8*k, -k], [-k, -k, -k]])
    img_sharpen = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    if show: show_img({"input":img, "sharpen": img_sharpen},cmap="gray")
    return img_sharpen

def opening(img, show=False):
    kernel1 = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img, kernel1, iterations = 2)
    kernel2 = np.ones((4,4), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel2, iterations = 2)
    if show: show_img({"input":img, "erosion":img_erosion, "dilation": img_dilation},cmap="gray")
    return img_dilation

def eight_bit_scaler(img, axis=None, show=False):
    min = img.min(axis=axis, keepdims=True)
    max = img.max(axis=axis, keepdims=True)
    img_min_max_scaled = 255 * (img-min)/(max-min)
    img_eight_bit_scaled = img_min_max_scaled.astype(np.uint8)
    if show: show_img({"input":img, "eight_bit_scaled": img_eight_bit_scaled},cmap="viridis")
    return img_eight_bit_scaled
    
def averaged_opening_data(img, show=False):
    C = -3.5
    binarized_img_cv_Cn3p5_w11 = cv2.adaptiveThreshold(img, 
                                                       255, 
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY, 
                                                       11, 
                                                       C)
    binarized_img_cv_Cn3p5_w21 = cv2.adaptiveThreshold(img,
                                                       255, 
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY, 
                                                       21, 
                                                       C)
    binarized_img_cv_Cn3p5_w31 = cv2.adaptiveThreshold(img,
                                                       255, 
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY, 
                                                       31, 
                                                       C)
    binarized_img_cv_Cn3p5_w41 = cv2.adaptiveThreshold(img,
                                                       255, 
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY, 
                                                       41, 
                                                       C)
    binarized_img_cv_Cn3p5_w51 = cv2.adaptiveThreshold(img,
                                                       255, 
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY, 
                                                       51, 
                                                       C)

    kernel = np.ones((3,3), np.uint8)
    opening_Cn3p5_w11 = cv2.morphologyEx(binarized_img_cv_Cn3p5_w11, cv2.MORPH_OPEN, kernel, iterations = 1)
    opening_Cn3p5_w21 = cv2.morphologyEx(binarized_img_cv_Cn3p5_w21, cv2.MORPH_OPEN, kernel, iterations = 1)
    opening_Cn3p5_w31 = cv2.morphologyEx(binarized_img_cv_Cn3p5_w31, cv2.MORPH_OPEN, kernel, iterations = 1)
    opening_Cn3p5_w41 = cv2.morphologyEx(binarized_img_cv_Cn3p5_w41, cv2.MORPH_OPEN, kernel, iterations = 1)
    opening_Cn3p5_w51 = cv2.morphologyEx(binarized_img_cv_Cn3p5_w51, cv2.MORPH_OPEN, kernel, iterations = 1)
    
    ave_opening_Cn3p5 = opening_Cn3p5_w11/5 + opening_Cn3p5_w21/5 + opening_Cn3p5_w31/5 + \
                    opening_Cn3p5_w41/5 + opening_Cn3p5_w51/5
    ave_opening_Cn3p5_bi = np.where(ave_opening_Cn3p5 > 0, 255, 0)

    if show:
        show_img({"input":img,
                  "binarized_Cn3p5_w11 ": binarized_img_cv_Cn3p5_w11,
                  "binarized_Cn3p5_w21 ": binarized_img_cv_Cn3p5_w21,
                  "binarized_Cn3p5_w31 ": binarized_img_cv_Cn3p5_w31,
                  "binarized_Cn3p5_w41 ": binarized_img_cv_Cn3p5_w41,
                  "binarized_Cn3p5_w51 ": binarized_img_cv_Cn3p5_w51,                       
                  },cmap="viridis")
        show_img({"input":img,
                  "opening_Cn3p5_w11 ": opening_Cn3p5_w11,
                  "opening_Cn3p5_w21 ": opening_Cn3p5_w21,
                  "opening_Cn3p5_w31 ": opening_Cn3p5_w31,
                  "opening_Cn3p5_w41 ": opening_Cn3p5_w41,
                  "opening_Cn3p5_w51 ": opening_Cn3p5_w51,
                  },cmap="viridis")
        show_img({"input":img,
                  "ave_Cn3p5": ave_opening_Cn3p5,
                  "ave_Cn3p5_bi": ave_opening_Cn3p5_bi,
                  },cmap="gray")
    return ave_opening_Cn3p5_bi

def apply_filters(img, thermal_npdat_path):
    lower_lim_pix_val, upper_lim_pix_val = get_limits(thermal_npdat_path)
    img_normed = normalize(img, lower_lim_pix_val, upper_lim_pix_val)
    img_clahe = clahe(img_normed)
    img_blured = blur(img_clahe)
    img_bilateral_filtered = bilateral_filter(img_blured)
    img_sharpen = sharpen(img_bilateral_filtered)
    img_opening = opening(img_sharpen)
    img_eight_bit_scaled = eight_bit_scaler(img_opening)
    img_averaged = averaged_opening_data(img_eight_bit_scaled)
    return img_averaged

def get_panel_contours(contours):
    panel_contours = []
    mu_list = []
    angle_list = []
    width_list = []
    height_list = []
    perimeter_list = []
    # 輪郭を１つずつ書き込んで出力
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if 50 <= area <= 500:
            peri = cv2.arcLength(contours[i], True)
            squareness = 4 * np.pi * area / peri**2
            # https://answers.opencv.org/question/171583/eliminate-unwanted-contours-opencv/            
            if 0.3 <= squareness <= 0.8:
                rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if 200 <= cv2.contourArea(box) <= 500:
                    info = get_rect_info(box)
                    panel_contours.append(box)
                    angle_list.append(info['angle'])
                    width_list.append(info['width'])
                    height_list.append(info['height'])
                    perimeter_list.append(peri)
    return panel_contours

def get_img_contours(img, panel_contours, color=False): 
    #img_con = cv2.drawContours(np.zeros_like(img_org), panel_contours, -1, 255, -1)
    img_con = cv2.fillPoly(np.zeros_like(img), panel_contours, 255)
    if color: img_con = cv2.cvtColor(img_con, cv2.COLOR_GRAY2BGR)
    return img_con

def add_index(img, panel_contours):
    if len(img.shape) < 3:
        img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)        
    else:
        img_colored = img;
    for i in range(len(panel_contours)):
        mu = cv2.moments(panel_contours[i])
        x, y = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
        cv2.putText(img_colored, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1)
    img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
    return img_colored

def fill_target_panels(img, target_contours):
    if len(img.shape) < 3:
        img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)        
    else:
        img_colored = img;
    img_filled = cv2.fillPoly(img_colored, target_contours, (0,255,255))
    return img_filled

def extract_modules(img, panel_contours):
    mult = 1.2   # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    img_box = img.copy()
    #img_box = cv2.cvtColor(im_con.copy(), cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(panel_contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit

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
            angle+=90
            rotated = True

        center = (int((x1+x2)/2), int((y1+y2)/2))
        size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
        #cv2.circle(img_box, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

        M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

        cropped = cv2.getRectSubPix(img_box, size, center)    
        cropped = cv2.warpAffine(cropped, M, size)

        croppedW = W if not rotated else H 
        croppedH = H if not rotated else W

        croppedRotated = cv2.getRectSubPix(cropped,
                                           (int(croppedW*mult), int(croppedH*mult)), (size[0]/2, size[1]/2))

        plt.imshow(croppedRotated,cmap='gray')
        cv2.imwrite("trimed/"+str(i)+".jpg", croppedRotated)
        plt.show()

        cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit
        plt.imshow(img_box,cmap='gray')
        plt.show()

if __name__ == "__main__":

    thermal_npdat_path = "./hokuto/thermal"
    img_path = './hokuto/thermal/DJI_0123_R.JPG'
    img_org = cv2.imread(img_path, 0)
    
    img_averaged = apply_filters(img_org, thermal_npdat_path)
    
    # 輪郭の検出
    contours, hierarchy = cv2.findContours(
        img_averaged.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # パネル情報の抽出
    panel_contours = get_panel_contours(contours)
    img_con = get_img_contours(img_org, panel_contours, index=True)
    img_index = add_index(img, panel_contours)
    show_img({"modules":img_con},cmap="gray")
    