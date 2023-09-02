from utils import cluster
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import warnings
from utils import tools
from utils import map_builder
from utils import interactional
from fcnn_gpu import train
from fcnn_gpu import predict

# 忽略所有警告
warnings.filterwarnings("ignore")

# 设置显示的最大行数和最大列数
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列


class Dataset_cluster(object):
    def __init__(self, dataset_folder):
        self.dataset = tools.load_csv(dataset_folder)
        self.metro_state = pd.read_csv(dataset_folder + "/" + "Metro.csv", encoding="gbk")[["gd经度", "gd纬度"]]
        self.uid_list = sorted(self.dataset["uid"].unique())
        self.uid_start = min(self.uid_list)
        self.uid_end = max(self.uid_list)
        self.our_uid = self.uid_list[201:301]
        self.uid_len = len(self.uid_list)
        print("数据集读取完成")
        self.index2label = {}
        self.label2index = {}
        self.cache = {}
        print("缓存建立完成")
        print("数据集中最大uid为{},最小uid为{}".format(self.uid_start, self.uid_end))
        print("我们要处理的uid范围为{} - {}".format(min(self.our_uid), max(self.our_uid)))
        # print("请输入你处理的数据范围,例如:0-20")
        # dl_st, dl_en = input().split("-")
        # self.dl_st = self.uid_list[int(dl_st)]
        # self.dl_en = self.uid_list[int(dl_en) + 1]

    def trajectory(self, uid):
        print("正在数据库中检索uid为{}的用户数据".format(uid))
        if uid not in self.uid_list:
            print("用户不存在")
            return None
        if uid in self.cache:
            print("已在缓存检索到该用户")
            return self.cache[uid]
        else:
            # selected_df = pd.read_excel("任务2-OD数据分析结果汇总表.xlsx", sheet_name="样例用户OD分段举例")
            selected_df = self.dataset[self.dataset['uid'] == uid]
            print("已在数据库中检索到该用户,正在进行数据清洗...")
            selected_df = selected_df[(selected_df != -1).all(axis=1) & ~selected_df.isna().any(axis=1)]
            selected_df['MedianTime'] = (selected_df['procedureEndTime'] + selected_df['procedureStartTime']) / 2
            sorted_df = selected_df.sort_values(by='MedianTime')
            for i in sorted_df.index.tolist():
                x, y = float(sorted_df.loc[i, 'longitude']), float(sorted_df.loc[i, 'latitude'])
                dy, dx = tools.delta(y, x)
                sorted_df.loc[i, 'longitude'], sorted_df.loc[i, 'latitude'] = dx + x, dy + y
            distances = cdist(
                sorted_df[['longitude', 'latitude']],
                self.metro_state[['gd经度', 'gd纬度']],
                metric='euclidean'
            )
            min_distances = np.min(distances, axis=1)
            sorted_df['metro_dist'] = np.where(min_distances > 0.002, 1, min_distances)
            sorted_df['metro_dist'] = tools.c_Sigmoid(sorted_df['metro_dist'])
            Cleandf = tools.DataCleaner(sorted_df)
            self.cache[uid] = Cleandf
            print("用户数据清洗完成,用户数据已缓存")
        return Cleandf

    def analysis(self, uid, config=None):
        res_dic = {}
        work = "不确定职业"
        df = self.trajectory(uid)
        data_analyst = cluster.Data_Ana(df, config)
        df = data_analyst.process()
        if config["标注模式"]:
            if config["映射"] == "人工":
                interacter = interactional.interact(df, config)
                res_dic, work = interacter.process()
            elif config["映射"] == "FCNN":
                df_t = df[config["五元组权重"].keys()].values
                res_dic, work = predict(data=df_t, config=config)
        if config["标注模式"]:
            idx = 0
            df_list = []
            for k, v in res_dic.items():
                if v["出行方式"] not in self.label2index:
                    self.label2index[v["出行方式"]] = idx
                    self.index2label[idx] = v["出行方式"]
                    idx += 1
            for k, v in res_dic.items():
                idx = self.label2index[v["出行方式"]]
                v["data"]["cluster"] = idx
                df_list.append(v["data"])
            df = pd.concat(df_list, ignore_index=False)
        self.build_map(df, uid)
        if config["标注模式"]:
            df = df.sort_values(by='procedureStartTime')
            st_df = df[["latitude", "longitude", "procedureStartTime", "procedureEndTime", "uid"]]
            st_df["划分OD段"] = df["cluster"]
            st_df["出行方式"] = df["cluster"].map(self.index2label)
            st_df["异常数据说明"] = df["异常数据说明"]
            tools.storage_csv("output/{}.csv".format(str(uid) + "-" + work), st_df)

    def build_map(self, df, uid):
        map = map_builder.ClusteredMap()
        map.add_trajectory_list(df, uid)
        map.save_map()
        map.open_map()


class Dataset_fcnn(object):
    def __init__(self, dataset_folder):
        self.dataset = tools.load_csv(dataset_folder)
        self.uid_list = sorted(self.dataset["uid"].unique())
        self.uid_start = min(self.uid_list)
        self.uid_end = max(self.uid_list)
        self.our_uid = self.uid_list[201:301]
        self.uid_len = len(self.uid_list)
        print("原数据集读取完成")

    def connection(self, uid, config=None):
        self.quintuple, find_flag = tools.search_quintuple_csv("quintuple", uid)
        self.quintuple.set_index(self.quintuple.columns[0], inplace=True)
        if not find_flag:
            print("不存在uid为{}的五元组".format(uid))
            return None
        self.output = tools.search_ouput_csv("output", uid)
        if self.output == None:
            print("不存在uid为{}的人工标注文件".format(uid))
        else:
            self.output = self.output[0]
        self.output.set_index(self.output.columns[0], inplace=True)
        common_index = self.quintuple.index.intersection(self.output.index)
        self.output = self.output.loc[common_index]
        t = self.output['出行方式'].tolist()
        t = [tools.invert_dict(config["出行方式映射"])[i.split(":")[0]] for i in t]
        self.output['出行方式'] = t

        self.output["latitude"] = 0
        self.output["longitude"] = 0
        self.output["MedianTime"] = 0
        self.output["metro_dist"] = 0
        self.output["speed"] = 0

        for i in common_index:
            try:
                self.output.loc[i, "latitude"] = self.quintuple[i, "latitude"]
                self.output.loc[i, "longitude"] = self.quintuple[i, "longitude"]
                self.output.loc[i, "MedianTime"] = self.quintuple[i, "MedianTime"]
                self.output.loc[i, "metro_dist"] = self.quintuple[i, "metro_dist"]
                self.output.loc[i, "speed"] = self.quintuple[i, "speed"]
            except:
                continue
        connected_list = np.array(self.output[list(config["五元组权重"].keys()) + ["出行方式"]].values.tolist())
        print("用户数据连接成功")
        return connected_list

# config = {"映射": "人工",
#           "手肘法线性映射": False,
#           "离散速度数量": 10,
#           "孤立森林异常比例": 0.1,
#           "最大簇数量": 13,
#           "五元组权重": {'latitude': 500, 'longitude': 500, 'MedianTime': 200, 'speed': 200,
#                          'metro_dist': 500},
#           "出行方式映射": {1: '驾驶', 2: '地铁', 3: '骑车', 4: '步行', 5: '高铁'}
#           }
# dd = Dataset_fcnn("data")
# data = dd.connection(171798692091, config=config)
# train(data, config=config)
