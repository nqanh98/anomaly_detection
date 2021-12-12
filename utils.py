
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(x, gamma=1, x_max=None):
    if x_max is None:
        x_max = np.max(x)
    else:
        x_max = x_max
    return np.clip(pow(x / x_max, gamma) * 255.0, 0, 255).astype(int)

def show_modules(img_dict, vmin=0, vmax=255):
    fig = plt.figure(figsize=(12,4),facecolor="w")
    n = len(img_dict)
    ax = {}
    for i, (k, v) in enumerate(img_dict.items()):
        ax[i] = fig.add_subplot(1,n,i+1)
        ax[i].imshow(v, vmin=vmin, vmax=vmax)
        ax[i].set_title(k)
    plt.show()
