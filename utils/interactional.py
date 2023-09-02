from . import tools
from . import map_builder
import matplotlib.pyplot as plt


class interact(object):
    def __init__(self, df, config=None):
        self.config = config
        self.dic = self.config["出行方式映射"]
        self.res_dic = {}
        print("进入交互程序")
        n_lable = df['cluster'].max() + 1
        self.df_list = []
        for i in range(n_lable):
            self.df_list.append(df[df['cluster'] == i])
        print("用户数据已按簇分离完成")

    def QA(self, df):
        record_dic = {}

        print("你认为这个簇的出行方式是什么?")
        for k, v in self.dic.items():
            print("{}.{}".format(k, v))
        idx_0 = int(input())
        record_dic["出行方式"] = self.dic[idx_0]
        if idx_0 == 2:
            print("请输入起始站:")
            m_st_0 = input()
            print("请输入终点站:")
            m_en_0 = input()
            record_dic["出行方式"] += ":{}-{}".format(m_st_0, m_en_0)
        elif idx_0 == 5:
            print("请输入火车站名称:")
            h_st_0 = input()
            record_dic["出行方式"] += ":从{}乘车出发".format(h_st_0)
        elif idx_0 == 4:
            print("请输入步行区域:")
            w_st_0 = input()
            record_dic["出行方式"] += ":{}".format(w_st_0)
        print("请输入异常数据说明(如果没有请输入“无特殊情况”或“无备注”)")
        record_dic["异常数据说明"] = input()
        df["异常数据说明"] = record_dic["异常数据说明"]
        record_dic['data'] = df
        print("对于此簇的信息是否确认无误并保存?")
        print("1.确认无误并保存")
        print("2.输入错误,请求重新输入")
        re_flag = input()
        if re_flag == "1":
            return True, record_dic
        else:
            return False, record_dic

    def process(self):
        print("进入交互环节")
        for index, df in enumerate(self.df_list):
            print("现在处理第{}个簇,共{}个簇".format(index + 1, len(self.df_list)))
            if 'iso_label' not in df.columns:
                df['iso_label'] = 0
                df['iso_label_t'] = 0

            while True:
                print("现在进行孤立森林离群点检测")
                df = tools.Iso_forest(df.copy(), self.config["孤立森林异常比例"])
                print("孤立森林异常点检测完成,生成地图及三维投影中")
                m = map_builder.ClusteredMap()
                m.add_trajectory_list(df=df, uid="tmp", color_label='iso_label_t')
                m.save_map()
                m.open_map()
                tools.tools_projection_multiple(df=df, args_list=['latitude', 'longitude', 'MedianTime', 'speed',
                                                                  'metro_dist'], label='iso_label_t')
                print("你认为这些点是异常点吗?\n1.是\n2.否")
                ac = input()
                if ac == "1":
                    for df_index in df.index.tolist():
                        if df.loc[df_index, 'iso_label_t'] == 1:
                            df.loc[df_index, 'iso_label'] = 1
                            df.loc[df_index, 'iso_label_t'] = -1
                elif ac == "2":
                    df = df[df["iso_label"] == 0]
                    print("异常点剔除完成,进入信息录入阶段")
                    while True:
                        re_flag, record = self.QA(df)
                        if re_flag:
                            self.res_dic[index] = record
                            break
                        else:
                            continue
                    break
        print("你认为他的职业是什么?例如:公司员工,外卖员,出租车司机,外来人员,不确定职业等")
        work = input()
        return self.res_dic, work
