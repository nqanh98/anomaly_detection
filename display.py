import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

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

def display_distributions(data, filepath="out.jpg", show=True, cluster_centers=None, gmm=None):
    fig = plt.figure(figsize=(12,4),facecolor="w")
    ax1 = fig.add_subplot(1,1,1)
    # -- distributions --
    for i in range(data.shape[1]):
        ax1.hist(data[:,i], bins = 100, alpha = 0.5)
    # -- mean and std -- 
    ax1.axvline(x = data.mean(), 
                color = 'green', 
                alpha = 0.8, 
                linestyle = '--', 
                label = 'Mean')
    ax1.axvline(x = data.mean() - 2*data.std(ddof=1), 
                color = 'orange', 
                alpha = 0.8, 
                linestyle = ':', 
                label = '2σ')
    ax1.axvline(x = data.mean() + 2*data.std(ddof=1), 
                color = 'orange', 
                alpha = 0.8, 
                linestyle = ':')
    ax1.axvline(x = data.mean() + 2*data.std(ddof=1), 
                color = 'orange', 
                alpha = 0.8, 
                linestyle = ':')
    ax1.axvline(x = np.percentile(data,25),
                color = 'blue', 
                alpha = 0.8, 
                linestyle = ':', 
                label = 'q1')
    ax1.axvline(x = np.percentile(data,50),
                color = 'blue', 
                alpha = 0.8, 
                linestyle = ':', 
                label = 'q2')
    ax1.axvline(x = np.percentile(data,75),
                color = 'blue', 
                alpha = 0.8, 
                linestyle = ':', 
                label = 'q3')
    # -- cluster centers --
    if cluster_centers is not None:
        for i in range(cluster_centers.shape[1]):
            for k in range(len(cluster_centers)):
                ax1.scatter(cluster_centers[k,i],-1)
                ax1.annotate(k,xy=(cluster_centers[k,i],-2))
    # -- config ax1 --
    ax1.set_title("Distribution of Thermal pixel values")
    ax1.set_xlabel('pxil values')
    ax1.set_ylabel('freq')
    ax1.legend(loc='upper left')
    #ax1.legend(loc='upper right')
    # -- gmm model --
    if gmm is not None:
        #x = np.linspace(0, 255, 300)
        x = np.linspace(data.min(), data.max(), 300)
        ax2 = ax1.twinx()
        for idx, c in gmm.index2class.items(): 
            gd = stats.norm.pdf(x, gmm.model.means_[idx, -1], np.sqrt(gmm.model.covariances_[idx]))
            ax2.plot(x, gmm.model.weights_[idx] * gd, label=c)
        # -- config ax2 --
        ax2.legend(loc='upper right')
        ax2.set_ylabel('prob')
    # -- outputs --
    plt.savefig(filepath)
    if show: 
        plt.show()
    else:
        plt.close()
        
def display_modules(img_dict, vmin=0, vmax=255):
    fig = plt.figure(figsize=(12,4),facecolor="w")
    n = len(img_dict)
    ax = {}
    for i, (k, v) in enumerate(img_dict.items()):
        ax[i] = fig.add_subplot(1,n,i+1)
        ax[i].imshow(v, vmin=vmin, vmax=vmax)
        ax[i].set_title(k)
    plt.show()

