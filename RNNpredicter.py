import pandas
import torch
from utils import tools
import numpy as np

import torch

# 设置PyTorch后端为CPU
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type(torch.FloatTensor)


class TimeSeriesPrediction:
    def __init__(self, user_id=171798692101, model_path='models/rnn.pth'):
        # 加载训练好的模型
        self.model = torch.load(model_path)

    def predict(self, input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # 使用模型进行预测
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # 提取接下来100个时间点的经纬度预测
        predicted_longitude = predictions[:, -100:, 0].numpy()
        predicted_latitude = predictions[:, -100:, 1].numpy()

        return predicted_longitude, predicted_latitude


def RNNpredicter(uid):
    quintuple, find_flag = tools.search_quintuple_csv(dataset_folder="quintuple", uid=uid)
    quintuple = quintuple[["latitude", "longitude", "MedianTime"]]
    data = np.array(quintuple.values.tolist())

    ts_prediction = TimeSeriesPrediction(user_id=uid, model_path='models/rnn.pth')
    predicted_longitude, predicted_latitude = ts_prediction.predict(data)
    # print(predicted_longitude)
    # data_map = pandas.DataFrame()