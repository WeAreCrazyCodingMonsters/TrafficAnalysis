import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import tools
import matplotlib.pyplot as plt  # 导入Matplotlib库
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNLocationPrediction:
    def __init__(self, data_folder, user_id, input_size=3, hidden_size=64, output_size=2, batch_size=32,
                 num_epochs=200):
        self.data_folder = data_folder
        self.user_id = user_id
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def prepare_data(self):
        quintuple, find_flag = tools.search_quintuple_csv(self.data_folder, self.user_id)
        quintuple = quintuple[["latitude", "longitude", "MedianTime"]]
        data = np.array(quintuple.values.tolist())

        # 划分数据集为训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[:, :3], data[:, :2], test_size=0.2,
                                                                                random_state=42)

        # 创建数据加载器
        train_dataset = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32).to(device),
                                      torch.tensor(self.y_train, dtype=torch.float32).to(device))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def build_model(self):
        self.model = self.RNNModel(self.input_size, self.hidden_size, self.output_size).to(device)

    def train_model(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # 添加一个列表来存储损失值
        loss_history = []

        for epoch in range(self.num_epochs):
            for batch_inputs, batch_targets in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

            # 记录损失值到历史列表
            loss_history.append(loss.item())
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item()}')

        torch.save(self.model, 'models/rnn.pth')
        # 绘制损失曲线
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig("fig/rnn_log.png")

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            test_inputs = torch.tensor(self.X_test, dtype=torch.float32)
            predictions = self.model(test_inputs)

    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNNLocationPrediction.RNNModel, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)  # 输出维度设置为2

        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out)  # 不需要降维
            return out


# 示例用法
def RNNTrainer(user_id=171798692101):
    data_folder = "quintuple"

    location_prediction = RNNLocationPrediction(data_folder, user_id)
    location_prediction.prepare_data()
    location_prediction.build_model()
    location_prediction.train_model()
    # location_prediction.evaluate_model()
