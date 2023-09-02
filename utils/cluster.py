from sklearn.preprocessing import MinMaxScaler
import numpy as np
from . import tools
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

import pandas as pd


class Data_Ana(object):
    def __init__(self, df, config=None):
        self.df = df
        self.config = config
        self.columns_to_normalize = config["五元组权重"]

    def bar(self, df):
        # 定义分割区间
        bins = list(range(0, 81, 10))
        # 将数据分配到分割区间中
        df['bin'] = pd.cut(df['speed'], bins=bins, right=False)
        # 计算每个分割区间的占比
        bin_counts = df['bin'].value_counts(normalize=True).sort_index()
        # 绘制柱状图
        plt.bar(bin_counts.index.astype(str), bin_counts.values)
        plt.show()

    def plot_elbow(self, n_components_range, bic_scores):
        # 手肘法曲线图
        plt.figure(figsize=(8, 6))
        plt.plot(n_components_range, bic_scores, marker='o')
        plt.title('Elbow Method for Gaussian Mixture')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC Score')
        plt.grid(True)
        plt.show()

    def plot_projection(self, n_list, labels):
        # 三维投影
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(n_list[:, 0], n_list[:, 1], n_list[:, 2], c=[tools.cluster_color[label] for label in labels],
                   marker='o',
                   label='inliers')
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.title("Data after Outlier Removal using Isolation Forest")
        plt.legend()
        plt.show()

    def normalize(self):
        # columns_to_normalize = ['diff_latitude', 'diff_longitude', 'diff_Time']

        # diff_Time = self.df['procedureEndTime'] - self.df['procedureStartTime']
        # self.df['diff_latitude'] = self.df['latitude'].diff()
        # self.df['diff_longitude'] = self.df['longitude'].diff()
        self.df['prev_time'] = self.df['MedianTime'].shift(1)
        self.df['time_diff'] = (self.df['MedianTime'] - self.df['prev_time']) / (1000 * 60 * 60)
        prev_latitude = self.df['latitude'].shift(1)
        prev_longitude = self.df['longitude'].shift(1)
        self.df['distance_km'] = tools.haversine_distance(prev_latitude, prev_longitude,
                                                          self.df['latitude'],
                                                          self.df['longitude'])
        self.df['speed'] = self.df['distance_km'] / self.df['time_diff']
        # print(df['speed'])
        self.df.loc[self.df.index[0], 'speed'] = 0
        # 每日时间重置
        # df['MedianTime'] = df['MedianTime'] % (1000 * 60 * 60 * 24)
        self.df['speed'] = self.df['speed'].apply(lambda x: min(x, 80))

        selected_columns_names = self.columns_to_normalize.keys()
        selected_columns_df = self.df[selected_columns_names]
        tools.storage_csv(path="quintuple/{}.csv".format(self.df['uid'].min()), df=selected_columns_df)
        df2list = selected_columns_df.values.tolist()
        df2array = np.array(df2list)

        outliers_indices = [1] * df2array.shape[0]
        # isolation_forest = IsolationForest(contamination=0.05)
        # isolation_forest.fit(df2array)
        # outliers_pred = isolation_forest.predict(df2array).astype(int)
        # outliers_indices = np.where(outliers_pred == -1)[0]
        # inliers = np.array(df2array[outliers_pred == 1])

        n_list = np.zeros((df2array.shape[0], 0))

        for index, k_v in enumerate(self.columns_to_normalize.items()):
            key, value = k_v
            scaler = MinMaxScaler(feature_range=(0, value))
            column_data = df2array[:, index].reshape(-1, 1)
            list_t = scaler.fit_transform(column_data)
            n_list = np.concatenate((n_list, list_t), axis=1)
        n_list = tools.speed_cluster(n_list, self.config["离散速度数量"])
        return n_list, outliers_indices

    def gaussian_mixture_cluster(self, n_list):
        n_components_range = range(1, self.config["最大簇数量"])
        bic_scores = []
        for n_components in n_components_range:
            gm = GaussianMixture(n_components=n_components, random_state=42)
            gm.fit(n_list.copy())
            bic_scores.append(gm.bic(n_list.copy()))
        if self.config["手肘法线性映射"]:
            n_cluster = tools.linear_mapping(np.argmin(bic_scores)) + 1
        else:
            n_cluster = np.argmin(bic_scores) + 1
        gm = GaussianMixture(n_components=n_cluster, random_state=42)
        gm.fit(n_list)
        labels = gm.fit_predict(n_list)

        # 手肘法绘图
        self.plot_elbow(n_components_range, bic_scores)

        # 三维投影
        # self.plot_projection(n_list,labels)
        return labels

    def process(self):
        n_list, outliers_indices = self.normalize()
        labels = self.gaussian_mixture_cluster(n_list)
        self.df['cluster'] = labels
        # interactional.interact(self.df,outliers_indices)
        return self.df
