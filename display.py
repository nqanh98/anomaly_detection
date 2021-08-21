import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np

def plot_2d_scatters_for_clusters(clusters, pred):
    # センターカラーの取得
    colors = []
    c_min = pred.cluster_centers_.min()
    c_max = pred.cluster_centers_.max()
    for n in range(pred.n_clusters):
        #color = cm.viridis(pred.cluster_centers_[n][0] / 255)
        color = cm.viridis((pred.cluster_centers_[n][0] - c_min) / (c_max - c_min))
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
    for n in range(pred.n_clusters):
        x = clusters[n].reshape(-1)
        y = np.zeros(len(x))
        ax.plot(x, y, "o", color=colors[n], ms=4, mew=0.5)
    plt.show()

def plot_3d_scatters_for_clusters(clusters, pred):
    # センターカラーの取得
    colors = []
    for n in range(pred.n_clusters):
        color = pred.cluster_centers_[n][:3] / 255
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
    for n in range(pred.n_clusters):
        ax.plot(clusters[n][:,0], clusters[n][:,1], clusters[n][:,2], "o", color=colors[n], ms=4, mew=0.5)
    plt.show()
