import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# 生成一些随机的点
n_samples = 300
n_features = 5
n_clusters = 2
random_state = 42

# 生成正态分布的点集
# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成第一个点集，较高密度
# 生成相邻点集的距离
distance_between_sets = 0.5

# 生成第一个点集，较高密度
num_points_high_density = 100
mean_high = [2, 2]
cov_high = [[0.05, 0], [0, 0.05]]
points_high_density = np.random.multivariate_normal(mean_high, cov_high, num_points_high_density)
x_coords = points_high_density[:, 0]
y_coords = points_high_density[:, 1]

left_top = (np.min(x_coords), np.max(y_coords))
right_bottom = (np.max(x_coords), np.min(y_coords))

# 生成第二个点集，较低密度
num_points_low_density = 100
mean_low = [mean_high[0] + distance_between_sets, mean_high[1]]
cov_low = [[0.4, 0], [0, 0.4]]
points_low_density = np.random.multivariate_normal(mean_low, cov_low, num_points_low_density)


import matplotlib.patches as patches
# left_top = (0.05, 0)
# right_bottom = (0, 0.05)
width = right_bottom[0] - left_top[0]
height = right_bottom[1] - left_top[1]



# 合并点集
X = np.vstack((points_high_density, points_low_density))

# 使用KMeans算法进行聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
kmeans_labels = kmeans.fit_predict(X)

# 使用DBSCAN算法进行聚类
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

GM = GaussianMixture(n_components=2)
GM_labels = GM.fit_predict(X)
# 绘制结果
plt.figure(figsize=(12, 5))

# 绘制KMeans结果
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red')
rect = patches.Rectangle(left_top, width, height, fill=False, color='r')
plt.gca().add_patch(rect)
plt.title('KMeans Clustering')

# 绘制DBSCAN结果
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
rect = patches.Rectangle(left_top, width, height, fill=False, color='r')
plt.gca().add_patch(rect)
plt.title('DBSCAN Clustering')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=GM_labels, cmap='viridis')
rect = patches.Rectangle(left_top, width, height, fill=False, color='r')
plt.gca().add_patch(rect)
plt.title('GM Clustering')


plt.tight_layout()
plt.show()