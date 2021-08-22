import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np

def plot_2d_scatters_for_clusters(clusters):
    # センターカラーの取得
    n_clusters = len(clusters)
    centers = np.stack([np.uint8(cluster.mean(axis=0)) for cluster in clusters])    
    c_min = centers.min()
    c_max = centers.max()
    colors = []
    for n in range(n_clusters):
        #color = cm.viridis(centers[n][0] / 255)
        color = cm.viridis((centers[n][0] - c_min) / (c_max - c_min))
        colors.append(color)
    # プロット
    fig = plt.figure(figsize=(6,6),facecolor="w")
    ax = fig.add_subplot(1, 1, 1)
    # 軸ラベルの設定
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    # 表示範囲の設定
    ax.set_xlim(0, 255)
    ax.set_ylim(-1, 1)
    for n in range(n_clusters):
        x = clusters[n].reshape(-1)
        y = np.zeros(len(x))
        ax.plot(x, y, "o", color=colors[n], ms=4, mew=0.5)
    plt.show()
    
def plot_3d_scatters_for_clusters(clusters):
    # センターカラーの取得
    n_clusters = len(clusters)
    colors = []
    for n in range(n_clusters):
        color = clusters[n].mean(axis=0) / 255
        colors.append(color)
    # プロット
    fig = plt.figure(figsize=(10,6),facecolor="w")
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # 軸ラベルの設定
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    # 表示範囲の設定
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    for n in range(n_clusters):
        ax.plot(clusters[n][:,0], clusters[n][:,1], clusters[n][:,2], "o", color=colors[n], ms=4, mew=0.5)
    plt.show()

def display_distributions(data, filepath="out.jpg", show=True, cluster_centers=None):
    fig = plt.figure(figsize=(12,6),facecolor="w")
    ax = fig.add_subplot(1,1,1)
    for i in range(data.shape[1]):
        ax.hist(data[:,i], bins = 100, alpha = 0.5)        
    ax.axvline(x = data.mean(), 
               color = 'green', 
               alpha = 0.8, 
               linestyle = '--', 
               label = 'Mean')
    ax.axvline(x = data.mean() - 2*data.std(ddof=1), 
               color = 'orange', 
               alpha = 0.8, 
               linestyle = ':', 
               label = '2σ')
    ax.axvline(x = data.mean() + 2*data.std(ddof=1), 
               color = 'orange', 
               alpha = 0.8, 
               linestyle = ':')
    if cluster_centers is not None:
        for i in range(cluster_centers.shape[1]):
            for k in range(len(cluster_centers)):
                ax.scatter(cluster_centers[k,i],-1)
                ax.annotate(k,xy=(cluster_centers[k,i],-2))
    ax.set_title("Distribution of Thermal pixel values")
    ax.set_xlabel('pxil values')
    ax.set_ylabel('freq')
    ax.legend(loc='upper right')
    plt.savefig(filepath)
    if show: 
        plt.show()
    else:
        plt.close()
