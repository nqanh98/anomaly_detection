
import numpy as np

def gamma_correction(x, gamma=1, x_max=None):
    if x_max is None:
        x_max = np.max(x)
    else:
        x_max = x_max
    return np.clip(pow(x / x_max, gamma) * 255.0, 0, 255).astype(int)

