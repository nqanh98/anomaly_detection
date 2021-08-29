import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

temperature_classes = ["High", "Medium", "Low", "Exception"]

class AnoGMM:
    def __init__(self, data):
        self.model = self.get_gmm_model(data)
        self.index2class = self.get_index2class()
        
    def get_gmm_model(self, data):
        model = GaussianMixture(
            n_components=len(temperature_classes),
            covariance_type='spherical'
        ).fit(data)
        return model

    def get_index2class(self):
        index2class = {
            idx:c for idx, c
            in zip(np.argsort(self.model.means_[:, -1])[::-1],temperature_classes)
        }
        return index2class

    def display(self):
        fig = plt.figure(figsize=(12,4),facecolor="w")
        ax1 = fig.add_subplot(1,1,1)
        #x = np.linspace(-6, 2, 300)
        x = np.linspace(0, 255, 300)
        gd = []
        for idx, c in self.index2class.items():
            gd = stats.norm.pdf(x, self.model.means_[idx, -1], np.sqrt(self.model.covariances_[idx]))
            plt.plot(x, self.model.weights_[idx] * gd, label=c)
        plt.legend()
        plt.show()

    def predict(self,data):
        return self.model.predict(data)
