from datasets import Dataset_cluster
from datasets import Dataset_fcnn
from fcnn_gpu import train
from master import Mission_1
import sys

ds = Dataset_cluster("data")
config = {"映射": "人工",  # 可选"人工","FCNN"
          "手肘法线性映射": False,  # 可选"True","False"
          "离散速度数量": 10,  # 可选[0,infinity]
          "孤立森林异常比例": 0.001,  # 可选[0,1]
          "最大簇数量": 5,  # 可选[1,infinity]
          "五元组权重": {'latitude': 500, 'longitude': 500, 'MedianTime': 200, 'speed': 200,
                         'metro_dist': 500},
          "出行方式映射": {1: '驾驶', 2: '地铁', 3: '骑车', 4: '步行', 5: '高铁'},
          "标注模式": True
          }

if __name__ == "__main__":
    if sys.argv[1] == 'cluster':
        if sys.argv[2] == 'label':
            config["映射"] = "人工"
        elif sys.argv[2] == 'FCNN':
            config["映射"] = "FCNN"
        ds = Dataset_cluster("data")
        while True:
            index = int(input("请输入你要处理的用户的uid:"))
            ds.analysis(index, config)
    elif sys.argv[1] == 'train':
        ds = Dataset_fcnn("data")
        data = ds.connection(171798692091, config=config)
        train(data, config=config)
    elif sys.argv[1] == 'Mission1':
        Mission_1()
