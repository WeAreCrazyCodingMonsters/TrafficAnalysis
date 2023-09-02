import pandas as pd
import webbrowser
from folium import Map, Marker, Popup, Icon, PolyLine
from datetime import datetime
from os import system

while (True):
    idx = int(input("请输入用户uid"))
    if idx == -1:
        break

    # 读取地图
    m = Map(location=[39.91, 116.40],
            zoom_start=12,
            tiles='http://wprd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}',
            attr='default')
    csv_path = r"metro_signalings_bj20230625\{}.csv".format(idx)

    # 读取用户文件
    try:
        data = pd.read_csv(csv_path)  # 读取文件
        system('cls')
        print("========现处理uid为{}的用户的文件========".format(idx))
    except:
        print("路径错误,请检查")
        continue
    trajectory_points = []
    last_point = [0, 0]
    avg_dist = 0
    lack_points_times = []

    # 将用户路径点添加到地图上
    for i in range(len(data)):
        x, y = float(data.loc[i, 'location'].split(',')[1]), float(data.loc[i, 'location'].split(',')[0])
        Marker(
            location=[x, y],
            popup=Popup("{}-{}".format(datetime.strptime(data.loc[i, 'start_time'], "%H:%M:%S").strftime("%H:%M")
                                       , datetime.strptime(data.loc[i, 'end_time'], "%H:%M:%S").strftime("%H:%M")),
                        parse_html=True,
                        max_width=100),
            icon=Icon(color='red' if data.loc[i, 'is_metro'] == 1 else 'blue')
        ).add_to(m)
        trajectory_points.append([data.loc[i, 'location'].split(',')[1], data.loc[i, 'location'].split(',')[0]])

        # 计算每两个路径点间的距离并添加到avg_dist上
        if i == 0:
            last_point = [x, y]
        else:
            dist = ((x - last_point[0]) ** 2 + (y - last_point[1]) ** 2) ** 0.5
            avg_dist += dist
            last_point = [x, y]
    tp = [[float(i[0]), float(i[1])] for i in trajectory_points]
    PolyLine(tp, color="green", opacity=0.7, weight=50).add_to(m)

    # 计算平均距离
    avg_dist /= len(data)

    last_point = [0, 0]

    # 当两个时间上相邻的路径点间的物理距离超过两点间平均距离的十倍时，会被本方法判定为距离过远从而用红色线段在地图上标注
    for i in trajectory_points:
        x, y = float(i[0]), float(i[1])
        if last_point == [0, 0]:
            last_point = [x, y]
            continue
        t_dist = ((x - last_point[0]) ** 2 + (y - last_point[1]) ** 2) ** 0.5
        if t_dist > 10 * avg_dist:
            filtered_df = data[data['location'] == i[1] + ',' + i[0]]

            PolyLine([last_point, i], color="red", opacity=1, weight=20).add_to(m)

        last_point = [x, y]

    m.save('NCUT.html')
    webbrowser.open('NCUT.html')

    print("请猜测用户出行方式:")
    print("1.驾驶")
    print("2.地铁")
    print("3.与地铁部分重合或其他")
    print("4.步行")
    print("5.不确定")
    idx_0 = int(input())
    if idx_0 == 2:
        print("请输入起始站:")
        m_st_0 = input()
        print("请输入终点站:")
        m_en_0 = input()

    print("是否存在信令丢失?")
    print("1.是")
    print("2.否")
    idx_1 = int(input())
    if idx_1 == 1:
        print("请输入丢失起始时间:")
        t_st_0 = input()
        print("请输入丢失终止时间:")
        t_en_0 = input()

    print("是否存在离群点?")
    print("1.是")
    print("2.否")
    idx_2 = int(input())

    xlsx_path = r"任务1-地铁轨迹数据标注 - 副本.csv"
    xlsx = pd.read_csv(xlsx_path, encoding="gbk")

    if idx_1 == 1:
        xlsx.loc[xlsx['UID'] == idx, '备注说明'] = "" if str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0])=='nan' else str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0]) + "存在信令丢失,"
        xlsx.loc[xlsx['UID'] == idx, '问题时段'] = "{}-{}".format(t_st_0, t_en_0)


    # print("" if str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0])=='nan' else str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0]))

    if idx_0 == 2:
        xlsx.loc[xlsx['UID'] == idx, 'is_metro_label'] = 1
        xlsx.loc[xlsx['UID'] == idx, '地铁站点'] = "{}-{}".format(m_st_0, m_en_0)
    elif idx_0 == 1:
        xlsx.loc[xlsx['UID'] == idx, 'is_metro_label'] = 0
        xlsx.loc[xlsx['UID'] == idx, '备注说明'] = "" if str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0])=='nan' else str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0]) + "存在长距离驾驶迹象,"
    elif idx_0 == 3:
        xlsx.loc[xlsx['UID'] == idx, 'is_metro_label'] = 0.5
        xlsx.loc[xlsx['UID'] == idx, '备注说明'] = "" if str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0])=='nan' else str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0]) + "路径与地铁线路部分重合,"
    elif idx_0 == 4:
        xlsx.loc[xlsx['UID'] == idx, 'is_metro_label'] = 0
        xlsx.loc[xlsx['UID'] == idx, '备注说明'] = "" if str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0])=='nan' else str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0]) + "存在步行迹象,"
    elif idx_0 == 5:
        xlsx.loc[xlsx['UID'] == idx, 'is_metro_label'] = 0

    if idx_2 == 1:
        xlsx.loc[xlsx['UID'] == idx, '备注说明'] = "" if str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0])=='nan' else str(xlsx.loc[xlsx['UID'] == idx]['备注说明'].values[0]) + "路径中存在离群点,"

    xlsx.to_csv("任务1-地铁轨迹数据标注 - 副本.csv", encoding="gbk", index=False)
    print("更改已保存!")
