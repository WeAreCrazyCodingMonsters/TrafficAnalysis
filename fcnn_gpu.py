from Network import NeuralNetwork
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def plot(epoch_num, loss_hist):
    # (655, 525)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.lineplot(x=[*range(1, epoch_num + 1)], y=loss_hist, ax=ax, label='loss')

    plt.savefig('fig/log.png', dpi=200)


def train(data, config=None):
    num_features = 5
    num_classes = 5
    X = data[:, :num_features]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 归一化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    categories_list = np.array(list(config["出行方式映射"].keys()))  # 将列表转换为NumPy数组
    # One-hot编码标签
    encoder = OneHotEncoder(sparse=False, categories=[categories_list])  # 传递一个包含数组的列表
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float32).to(device)

    model = NeuralNetwork(num_features=num_features, num_classes=num_classes).to(device)

    summary(model, input_size=(100, 5))

    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    batch_size = 32

    losses = []
    for epoch in range(num_epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
    plot(loss_hist=losses, epoch_num=num_epochs)
    eval(model=model, X_test_tensor=X_test_tensor, criterion=criterion, y_test_tensor=y_test_tensor)
    torch.save(model, 'models/fcnn.pth')
    # 做出预测
    predicted_probs = model(X_test_tensor)
    # predicted_labels = torch.argmax(predicted_probs, dim=1)
    predicted_labels = encoder.inverse_transform(predicted_probs.detach().numpy())


def predict(data, config):
    data = np.array(data)
    num_features = 5
    num_classes = 5
    X = data[:, :num_features]

    y = [[1], [2], [3], [4], [5]]
    # 创建并拟合编码器
    categories_list = np.array(list(config["出行方式映射"].keys()))  # 将列表转换为NumPy数组
    encoder = OneHotEncoder(sparse=False, categories=[categories_list])  # 传递一个包含数组的列表
    encoder.fit(y)

    model = torch.load('models/fcnn.pth')
    model.eval()  # 设置模型为评估模式，不启用梯度计算
    X_tensor = torch.tensor(X, dtype=torch.float32)

    predicted_probs = model(X_tensor)

    predicted_labels = (encoder.inverse_transform(predicted_probs.detach().numpy())).reshape(-1)
    return predicted_labels, "不确定职业"


def eval(model, X_test_tensor, criterion, y_test_tensor):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

# 读取DataFrame数据
# 请将以下部分替换为你自己的数据读取代码
# df = pd.read_csv('your_data.csv')

# 假设你已经从DataFrame中提取了特征和标签
# X = df[['latitude', 'longitude', 'MedianTime', 'speed', 'metro_dist']].values
# y = df['label'].values

# 假设有5个特征和5个输出类别


# 划分训练集和测试集


# 在测试集上评估模型
