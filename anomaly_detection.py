import numpy as np
from sklearn.mixture import GaussianMixture

temperature_classes = ["High", "Medium", "Low", "Exception"]

def get_gmm(data):
    gmm = GaussianMixture(
        n_components=len(temperature_classes),
        covariance_type='spherical'
    ).fit(data)
    return gmm

def get_index2class(model):
    index2class = {
        idx:c for idx, c
        in zip(np.argsort(model.means_[:, -1])[::-1],temperature_classes)
    }
    return index2class


