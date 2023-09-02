import webbrowser
from folium import CircleMarker, Popup, PolyLine
from . import tools
import seaborn as sns
import os
from folium import Map, Marker


class ClusteredMap(object):
    def __init__(self):
        self.m = Map(location=[39.91, 116.40],
                     zoom_start=12,
                     tiles='http://wprd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}',
                     attr='default')
        self.uid = -1

    def add_trajectory_list(self, df, uid, color_label='cluster'):
        self.uid = uid
        # inliers, outliers_indices = cluster.normalize(df.copy())
        # df['cluster'] = cluster.GaussianMixtureCluster(inliers.copy())

        # cluster_data_ana = cluster.Data_Ana(df.copy())
        # df['cluster'], outliers_indices = cluster_data_ana.process()

        cluster_palette = sns.color_palette("Dark2")

        # for i in metro_state.index.tolist():
        #     x, y = float(metro_state.loc[i, 'gd经度']), float(metro_state.loc[i, 'gd纬度'])
        #     marker = Marker(location=[y, x])  # 注意这里的坐标顺序
        #     marker.add_to(self.m)

        for i in df.index.tolist():
            x, y = float(df.loc[i, 'latitude']), float(df.loc[i, 'longitude'])
            cluster_idx = int(df.loc[i, color_label])
            if cluster_idx == -1:
                continue
            cluster_color = cluster_palette[cluster_idx]
            cluster_color = "#{:02X}{:02X}{:02X}".format(
                int(cluster_color[0] * 255), int(cluster_color[1] * 255), int(cluster_color[2] * 255)
            )
            CircleMarker(
                location=(x, y),
                popup=Popup(
                    "{}-{}".format(tools.HMSnt2HM(tools.ms2nt(df.loc[i, 'procedureStartTime'])),
                                   tools.HMSnt2HM(tools.ms2nt(df.loc[i, 'procedureEndTime']))),
                    parse_html=True,
                    max_width=100),
                radius=6,
                color=cluster_color,
                fill_color=cluster_color,  # 填充颜色
                fill_opacity=1,  # 填充不透明度
                fill=True).add_to(self.m)

        # self.avg_dist = self.calculate_avg_distance(df)

    #
    # def calculate_avg_distance(self, df):
    #     avg_dist = 0
    #     last_point = None
    #     for i in df.index.tolist():
    #         x, y = float(df.loc[i, 'latitude']), float(df.loc[i, 'longitude'])
    #         if last_point is not None:
    #             dist = ((x - last_point[0]) ** 2 + (y - last_point[1]) ** 2) ** 0.5
    #             avg_dist += dist
    #         last_point = (x, y)
    #     avg_dist /= len(df)
    #     return avg_dist
    #
    # def long_distance_poly(self):
    #     last_point = None
    #     for i in self.trajectory_points:
    #         x, y = float(i[0]), float(i[1])
    #         if last_point is not None:
    #             t_dist = ((x - last_point[0]) ** 2 + (y - last_point[1]) ** 2) ** 0.5
    #             if t_dist > 10 * self.avg_dist:
    #                 PolyLine([last_point, i], color="red", opacity=1, weight=20).add_to(self.m)
    #         last_point = (x, y)

    def save_map(self):
        self.m.save("htmls/{}.html".format(str(self.uid)))

    def open_map(self):
        webbrowser.open(str(os.getcwd()) + '/'+ "htmls/{}.html".format(str(self.uid)))
