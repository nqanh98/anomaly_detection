import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def eight_bit_scaler(x, axis=None):
  min = x.min(axis=axis, keepdims=True)
  max = x.max(axis=axis, keepdims=True)
  min_max_scaled = (x-min)/(max-min)
  pre_eight_bit_scaled = 255 * min_max_scaled
  result = pre_eight_bit_scaled.astype(np.uint8)
  return result

def averaged_adaptive_threshold(gamma_corrected_8bit, window_size_list, C=-7):
    img_arr = np.zeros((len(window_size_list), gamma_corrected_8bit.shape[0], gamma_corrected_8bit.shape[1]))
  
    for i, window_size in enumerate(window_size_list):
        binarized = cv2.adaptiveThreshold(gamma_corrected_8bit, 
                                          255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 
                                          window_size, 
                                          C)
        img_arr[i, :, :] = binarized

        # result
        plt.figure(figsize=(20, 20),facecolor="w")
        plt.subplot(1, 2, 1)
        plt.imshow(gamma_corrected_8bit,
                   interpolation = 'nearest', 
                   cmap = 'viridis')
        plt.title('Before')
        plt.subplot(1, 2, 2)
        plt.imshow(binarized,
                   interpolation = 'nearest', 
                   cmap = 'viridis')
        plt.title('Window size = ' + str(window_size))
        plt.show()

    return np.mean(img_arr, axis=0)

def preprocessing(org_img, 
                  upper_lim_pix_val, lower_lim_pix_val, 
                  flip_flag=False, 
                  clipLimit=2.0, tileGridSize=(8, 8), 
                  blur_kernel_size=(3, 3), 
                  bilateral_d=3, bilateral_sigmaColor=200, bilateral_sigmaSpace=50, 
                  sharp_kernel_value=1.0, 
                  inflate_flag=True,
                  gamma = 4.8, 
                  window_size_list=[11, 21, 31, 41, 51, 61, 71],
                  C=-7, 
                  ensemble_flag=True):

    # flip the values
    if flip_flag:
        flipped_img = cv2.bitwise_not(org_img)
    else:
        flipped_img = org_img

    # normalize
    temp = np.where(flipped_img > upper_lim_pix_val, upper_lim_pix_val, flipped_img)
    normed_img = np.where(temp < lower_lim_pix_val, lower_lim_pix_val, temp)

    del temp

    # contrast
    # clipLimit: If you want to enhance edges, you should raise the value
    normed_img_8 = np.array(normed_img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl1 = clahe.apply(normed_img_8)

    del normed_img

    plt.figure(figsize=(10, 6),facecolor="w")

    plt.subplot(1, 2, 1)
    plt.imshow(normed_img_8,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(cl1,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('After')
    
    plt.suptitle('Contrast', fontsize=10)
    plt.show()

    # blur
    blured_img = cv2.blur(cl1, blur_kernel_size)

    del normed_img_8

    plt.figure(figsize=(10, 6),facecolor="w")

    plt.subplot(1, 2, 1)
    plt.imshow(cl1,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('Before')
    
    plt.subplot(1, 2, 2)
    plt.imshow(blured_img,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('After')
    
    plt.suptitle('Blur', fontsize=10)
    plt.show()
    
    del cl1

    # bilateral filter
    # bilateral_sigmaColor: If bilateral_sigmaColor is large, a large weight is adopted even if the difference in pixel values ​​is large.
    # bilateral_sigmaSpace: If bilateral_sigmaSpace is large, a large weight is adopted even if the distance between pixels is wide.
    bilateral_filtered_img = cv2.bilateralFilter(blured_img.astype(np.float32), bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)

    plt.figure(figsize=(10, 6),facecolor="w")
    
    plt.subplot(1, 2, 1)
    plt.imshow(blured_img,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('Before')
    
    plt.subplot(1, 2, 2)
    plt.imshow(bilateral_filtered_img,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('After')
    
    plt.suptitle('Bilateral', fontsize=10)
    plt.show()
    
    del blured_img

    # Sharpening
    # sharp_kernel_value: If you want to enhance edges, you should raise the value
    sharpening_kernel = np.array([[-sharp_kernel_value, -sharp_kernel_value, -sharp_kernel_value], 
                                  [-sharp_kernel_value, 1+8*sharp_kernel_value, -sharp_kernel_value], 
                                  [-sharp_kernel_value, -sharp_kernel_value, -sharp_kernel_value]])
    sharpen = cv2.filter2D(bilateral_filtered_img, ddepth=-1, kernel=sharpening_kernel)
    plt.figure(facecolor="w")
    plt.imshow(sharpen, cmap = "gray")
    plt.title('Sharpening')
    plt.show()
    
    # Inflate the edge
    if inflate_flag:
        ## Erosion
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(sharpen, kernel, iterations = 2)
        plt.figure(facecolor="w")
        plt.imshow(erosion, cmap = "gray")
        plt.title('Erosion')
        plt.show()
        
        ## Dilation
        kernel = np.ones((4,4), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations = 2)
        plt.figure(facecolor="w")
        plt.imshow(dilation, cmap = "gray")
        plt.title('Dilation')
        plt.show()
    else:
        dilation = sharpen

    #　Gamma correction
    # gamma: If you want to enhance edges, you should raise the value.
    #          However, modules may be separated due to clustering of pixel values (gamma correction.)
    dilation_8bit = eight_bit_scaler(dilation)

    del dilation, erosion

    gamma_corrected = np.power(dilation_8bit / float(np.max(dilation_8bit)), gamma)
    gamma_corrected_8bit = eight_bit_scaler(gamma_corrected)
    plt.figure(figsize=(10, 6),facecolor="w")
    plt.subplot(1, 2, 1)
    plt.imshow(dilation_8bit,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('Before')
    plt.subplot(1, 2, 2)
    plt.imshow(gamma_corrected_8bit,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.title('After')
  
    plt.suptitle('Gamma correction', fontsize=10)
    plt.show()

    # Binarized
    ave = averaged_adaptive_threshold(gamma_corrected_8bit, window_size_list, C=-7)
    
    del gamma_corrected, gamma_corrected_8bit
    
    plt.figure(facecolor="w")
    plt.imshow(ave,
               interpolation = 'nearest', 
               cmap = 'gray')
    plt.colorbar()
    plt.title('AVE')
    plt.show()
    
    if ensemble_flag:
        ave_bi = np.where(ave > 255/2, 255, 0)
        
        plt.figure(facecolor="w")
        plt.imshow(ave_bi,
                   interpolation = 'nearest', 
                   cmap = 'gray')
        plt.title('bi')
        plt.show()
    else:
        ave_bi = np.where(ave > 0, 255, 0)
        
        plt.figure(facecolor="w")
        plt.imshow(ave_bi,
                   interpolation = 'nearest', 
                   cmap = 'gray')
        plt.title('bi')
        plt.show()

    return ave_bi

def select_contours_and_minAreaRect(img_binarized, 
                                    area_min=50, area_max=3000, 
                                    aspect_min=0.5, aspect_max=1.0):
  # 輪郭の検出
  contours, hierarchy = cv2.findContours(img_binarized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  im_con = cv2.drawContours(np.zeros_like(img_binarized.astype(np.uint8)), contours, -1, 255, -1)
  plt.figure(figsize=(10, 5),facecolor="w")
  plt.subplot(1, 1, 1)
  plt.imshow(im_con, cmap='gray')
  plt.title('Contours')
  plt.show()

  panel_contours = []

  # 輪郭を１つずつ書き込んで出力
  for i in range(len(contours)):
      area = cv2.contourArea(contours[i])
      if area_min <= area <= area_max:
        # peri = cv2.arcLength(contours[i], True)
        # squareness = 4 * np.pi * area / peri**2
        # if 0.3 <= squareness <= 0.8: # https://answers.opencv.org/question/171583/eliminate-unwanted-contours-opencv/
        rect = cv2.minAreaRect(contours[i])

        (x, y), (width, height), angle = rect
        aspect_ratio = min(width, height) / max(width, height)

        if aspect_min <= aspect_ratio <= aspect_max:
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          
          if area_max/3 <= cv2.contourArea(box) <= area_max:
            panel_contours.append(box)

  return panel_contours
