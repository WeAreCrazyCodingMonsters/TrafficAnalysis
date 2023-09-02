import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest

# 生成一些带有离群点的示例数据
np.random.seed(0)
n_samples = 200
n_outliers = 10

X = 0.3 * np.random.randn(n_samples - n_outliers, 3)
X = np.r_[X + 2, X - 2, np.random.uniform(low=-6, high=6, size=(n_outliers, 3))]

# 绘制原始数据
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='o', label='inliers')
plt.title("Original Data")
plt.legend()
plt.show()

# 使用孤立森林方法去除离群点
isolation_forest = IsolationForest(contamination=0.05)
isolation_forest.fit(X)
outliers_pred = isolation_forest.predict(X)
inliers = X[outliers_pred == 1]
outliers_if = X[outliers_pred == -1]

# 绘制去除离群点后的数据
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c='b', marker='o', label='inliers')
ax.scatter(outliers_if[:, 0], outliers_if[:, 1], outliers_if[:, 2], c='r', marker='x', label='outliers')
plt.title("Data after Outlier Removal using Isolation Forest")
plt.legend()
plt.show()
