
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(x, gamma=1, x_max=None):
    if x_max is None:
        x_max = np.max(x)
    else:
        x_max = x_max
    return np.clip(pow(x / x_max, gamma) * 255.0, 0, 255).astype(int)

def show_img(img_dict, vmin=0, vmax=255, cmap=None, figsize=(12,4)):
    fig = plt.figure(figsize=(12,4),facecolor="w")
    n = len(img_dict)
    ax = {}
    for i, (k, v) in enumerate(img_dict.items()):
        ax[i] = fig.add_subplot(1,n,i+1)
        if cmap is not None:
            ax[i].imshow(v, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            ax[i].imshow(v, vmin=vmin, vmax=vmax)
        ax[i].set_title(k)
    plt.show()

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
