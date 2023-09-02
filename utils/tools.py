import pandas as pd
import glob
from datetime import datetime
import math
import transbigdata
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import IsolationForest

cluster_palette = sns.color_palette("Dark2")
cluster_color = ["#{:02X}{:02X}{:02X}".format(
    int(i[0] * 255), int(i[1] * 255), int(i[2] * 255)
) for i in cluster_palette]


def f1(x):
    return np.exp(1 * (-x + 0))


def f2(x):
    return (1 / (1 + f1(x)))


def f3(x):
    return 1 - f2((10 / 0.003) * (x - 0.0035)) / 1


def c_Sigmoid(x):
    y = f3(x)
    return y


def load_csv(dataset_folder):
    csv_files = glob.glob("{}/*.snappy.csv".format(dataset_folder))  # 修改为包含CSV文件的文件夹路径
    dfs = []
    # 循环读取并添加DataFrame到列表中
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=False)
    return combined_df


def search_quintuple_csv(dataset_folder, uid):
    uid = str(uid)
    found_files = []
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            # 使用os.path.splitext分离文件名和扩展名
            filename, file_extension = os.path.splitext(file)

            # 检查文件名是否匹配
            if filename == uid:
                found_files.append(os.path.join(root, file))

    if found_files:
        csv_file_path = found_files[0]
        df = pd.read_csv(csv_file_path, encoding="gbk")
        return df, True
    else:
        return None, False


def search_ouput_csv(folder_path, target_uid):
    target_uid = str(target_uid)
    matching_files = []

    for filename in os.listdir(folder_path):
        if filename.startswith(target_uid + "-") and filename.endswith(".csv"):
            matching_files.append(os.path.join(folder_path, filename))

    if not matching_files:
        print("未找到匹配的文件.")
        return None

    dataframes = []

    for file_path in matching_files:
        df = pd.read_csv(file_path, encoding="gbk")
        dataframes.append(df)

    return dataframes


def storage_csv(path, df):
    df.to_csv(path, index=True, header=True, encoding="gbk")


def ms2nt(ms):
    seconds = int(ms / 1000)
    nt = datetime.fromtimestamp(seconds)
    return nt


def HMSnt2HM(hms):
    hm = hms.strftime("%Y-%m-%d %H:%M")
    return hm


import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # 计算差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine公式计算距离
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance


def linear_mapping(x):
    a = 5 / 11
    b = -5 / 11
    y = a * x + b
    return int(y)


def DataCleaner(df):
    df['MedianTime_'] = df['MedianTime']
    df = transbigdata.traj_clean_redundant(data=df, col=['uid', 'MedianTime_', 'longitude', 'latitude'])
    df = transbigdata.traj_clean_drift(data=df, col=['uid', 'MedianTime_', 'longitude', 'latitude'])
    # df = transbigdata.traj_smooth(data=df, col=['uid', 'MedianTime_', 'longitude', 'latitude'])
    # df = transbigdata.traj_sparsify(data=df, col=['uid', 'MedianTime_', 'longitude', 'latitude'])
    return df


from sklearn.cluster import KMeans


def speed_cluster(data, cluster_num):
    # 创建KMeans模型实例
    kmeans = KMeans(n_init='auto', n_clusters=cluster_num)

    # 对指定列进行训练，reshape是为了确保输入的是二维数组
    column_to_cluster = data[:, 3].reshape(-1, 1)
    kmeans.fit(column_to_cluster)

    # 获取簇心并写回数组
    cluster_centers = kmeans.cluster_centers_

    # 更新数据数组中指定列的值为对应簇的中心值
    data[:, 3] = cluster_centers[kmeans.labels_].flatten()

    # 打印簇心
    # print("Cluster Centers:", cluster_centers)

    return data


def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(math.fabs(x))
    ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(math.fabs(x))
    ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0
    return ret


PI = 3.14159265358979324
x_pi = 3.14159265358979324 * 3000.0 / 180.0


def delta(lat, lon):
    a = 6378245.0
    ee = 0.00669342162296594323
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * PI
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI)
    return dLat, dLon


def tools_projection_single(df, args_list, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[args_list[0]], df[args_list[1]], df[args_list[2]], c=[cluster_color[label] for label in labels],
               marker='o',
               label='Point Set')
    plt.xlabel(args_list[0])
    plt.ylabel(args_list[1])
    plt.title("x-{},y-{},z-{}".format(args_list[0], args_list[1], args_list[2]))
    plt.legend()
    plt.show()


def Iso_forest(dataframe, contamination_ratio):
    # contamination = 0.1
    clf = IsolationForest(contamination=contamination_ratio, random_state=42)
    t_df = dataframe[dataframe["iso_label"] == 0]
    median_time_array = t_df[['MedianTime']]
    clf.fit(median_time_array)
    iso_res = clf.predict(median_time_array)
    predicted_indexes = np.where(iso_res == -1)[0]
    predicted_df_indexes = t_df.iloc[predicted_indexes].index
    for i in predicted_df_indexes:
        dataframe.loc[i, "iso_label_t"] = 1
    return dataframe


def tools_projection_multiple(df, args_list, label):
    outlier = df[df[label] == 1].index
    inliers = df[df[label] == 0].index
    fig = plt.figure()
    fig.set_size_inches(16, 8)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(df.loc[inliers, args_list[0]], df.loc[inliers, args_list[1]], df.loc[inliers, args_list[2]],
                c='b', marker='o', label='inliers')
    ax1.scatter(df.loc[outlier, args_list[0]], df.loc[outlier, args_list[1]], df.loc[outlier, args_list[2]],
                c='r', marker='x', label='outliers')
    ax1.set_title("x-{},y-{},z-{}".format(args_list[0], args_list[1], args_list[2]))

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(df.loc[inliers, args_list[0]], df.loc[inliers, args_list[1]], df.loc[inliers, args_list[3]],
                c='b', marker='o', label='inliers')
    ax2.scatter(df.loc[outlier, args_list[0]], df.loc[outlier, args_list[1]], df.loc[outlier, args_list[3]],
                c='r', marker='x', label='outliers')
    ax2.set_title("x-{},y-{},z-{}".format(args_list[0], args_list[1], args_list[3]))

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(df.loc[inliers, args_list[0]], df.loc[inliers, args_list[1]], df.loc[inliers, args_list[4]],
                c='b', marker='o', label='inliers')
    ax3.scatter(df.loc[outlier, args_list[0]], df.loc[outlier, args_list[1]], df.loc[outlier, args_list[4]],
                c='r', marker='x', label='outliers')
    ax3.set_title("x-{},y-{},z-{}".format(args_list[0], args_list[1], args_list[4]))
    plt.tight_layout()
    plt.show()


def invert_dict(original_dict):
    inverted_dict = {}
    for key, value in original_dict.items():
        inverted_dict[value] = key
    return inverted_dict
