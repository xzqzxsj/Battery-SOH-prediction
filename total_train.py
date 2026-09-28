import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import copy
from MAgent.Mamba_Agent import Mamba_Agent
from MAgent.MAgent_AWA import MambaAgentWithFedAWA
from sklearn.metrics import r2_score

def mape_func(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


# 自定义Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 100个窗口去预测后 1 个
def prepare_multi_timestep_data(all_veh_data, window_size=100, forecast_horizon=1, scaler=None):
    all_X, all_y = [], []
    if scaler is None:
        all_data = np.vstack(all_veh_data)
        max_abs_values = np.max(np.abs(all_data), axis=0)
        # 避免除以零
        max_abs_values[max_abs_values == 0] = 1
        scaler = max_abs_values
    else:
        max_abs_values = scaler
    for car_data in all_veh_data:
        # 标准化数据
        normalized_data = car_data / max_abs_values
        n_timesteps = normalized_data.shape[0]
        for i in range(0, n_timesteps - window_size - forecast_horizon + 1):
            # 输入窗口: 所有特征 + 容量
            X_window = normalized_data[i: i + window_size, 1:]  # 所有特征包括容量
            # 输出: 未来10个周期的容量, 最后一列是容量
            y_window = normalized_data[i + window_size: i + window_size + forecast_horizon, -1]
            all_X.append(X_window)
            all_y.append(y_window)
    return np.array(all_X), np.array(all_y), scaler


# 训练过程
def train():
    epochs = 150
    # 损失函数和优化器
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("Starting training...")
    for epo in range(epochs):
        model.train()
        train_mae, train_mse, train_rmse = 0, 0, 0
        for i, (features, capacity) in enumerate(train_loader):
            # features=features.unsqueeze(2)
            features = features.to(device)
            capacity = capacity.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = mse_loss(outputs, capacity)  # 用均方误差作为损失函数
            mae = mae_loss(outputs, capacity)
            rmse = torch.sqrt(loss)

            train_mae+=mae.item()
            train_mse+=loss.item()
            train_rmse+=rmse.item()

            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 计算平均情况
        train_mae=train_mae/len(train_loader)
        train_mse = train_mse / len(train_loader)
        train_rmse = train_rmse / len(train_loader)
        print("epo:{},mae:{},mse:{},rmse:{}".format(epo+1, train_mae, train_mse, train_rmse))


# 测试过程
def test(test_loader,number):  # 输入4个测试集的其中之一
    print("Starting testing...")
    capacity_scaler = scaler[-1]  # 容量列的标准化因子
    # 损失函数
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    test_mae = []
    test_mse = []
    test_rmse = []
    test_mape=[]

    predict=[]   # 记录预测容量
    real=[]    # 记录真实容量

    with torch.no_grad():
        model.eval()

        for j, (features, capacity) in enumerate(test_loader):
            features = features.to(device)
            # features = features.unsqueeze(2)
            capacity = capacity.to(device)   # 真实值
            outputs = model(features)  # 预测值

            test_mse.append(mse_loss(outputs, capacity).item())
            test_mae.append(mae_loss(outputs, capacity).item())
            mse=mse_loss(outputs, capacity)
            test_rmse.append(torch.sqrt(mse).item())

            # 将张量移动到 CPU 并转换为 NumPy 数组
            outputs_np = outputs.cpu().numpy()
            capacity_np = capacity.cpu().numpy()
            mape=mape_func(capacity_np,outputs_np)
            test_mape.append(mape)

            outputs_restored = outputs_np * capacity_scaler
            capacity_restored = capacity_np * capacity_scaler

            # 确保数据是一维的
            predict.extend(outputs_restored.flatten().tolist())  # 展平并添加到列表
            real.extend(capacity_restored.flatten().tolist())  # 展平并添加到列表

        # 统计单辆车的损失
        mean_mse = np.mean(test_mse)
        mean_mae = np.mean(test_mae)
        mean_rmse = np.mean(test_rmse)
        mean_r2=r2_score(real,predict)
        mean_mape=np.mean(test_mape)

        print('测试平均MAE', mean_mae)
        print('测试平均MSE', mean_mse)
        print('测试平均RMSE', mean_rmse)
        print('测试平均R2',mean_r2)
        print('测试平均MAPE',mean_mape)

    # 画出单辆车的预测曲线和真实曲线
    # 由于预测数量为 600多个周期，所以划分成多块展示，每块100个，最后一块为剩余周期
    plt.figure(figsize=(16, 20))
    k=len(predict)
    for i in range(0,k,100):
        pre=predict[i:min(i+100,k)]
        refer=real[i:min(i+100,k)]
        plt.plot(pre,'ro-', markersize=2, label='Model')
        plt.plot(refer, 'bo-', markersize=2, label='Real')
        plt.legend()
        plt.ylabel('SOH')
        plt.xlabel('Cycle')
        plt.show()

    # 最后总体拼接查看
    total_ca=veh_ca[number]
    plt.figure(figsize=(16, 20))
    plt.title('Test')
    plt.plot(total_ca, 'ro-', markersize=2, label='Real')
    plt.plot(np.arange(100, len(total_ca), 1), predict,'bo-', markersize=2, label='Model')
    plt.legend()
    plt.axvline(x=100, c='k', ls='--')  # 添加垂直线
    plt.title('Veh {}'.format(car_id[number]))
    plt.ylabel('SOH')
    plt.xlabel('Cycle')
    plt.show()

    df=pd.DataFrame(np.array(predict),columns=['Prediction SOH'])
    df.to_csv(r"C:\Users\lenovo\OneDrive\桌面\#177.csv")

if __name__ == "__main__":
    N = 7
    T = 100
    # 数据加载和预处理-- 这里替换成你的数据和车辆编号
    dir_path = r"E:\My_EV_set_0\SOH_data"
    car_id = [i for i in range(1, 21)]
    
    all_veh_data = []  # 存放所有车的特征数据
    veh_ca = []

    for i in range(len(car_id)):
        id=car_id[i]
        path = dir_path + "\#{}.csv".format(id)
        veh = pd.read_csv(path)
        data = copy.deepcopy(veh.values)  # 深拷贝
        ca = copy.deepcopy(veh.values[:, -1])  # 最后一列为容量
        all_veh_data.append(np.array(data))
        veh_ca.append(list(ca))

    # 准备训练和测试数据
    all_train_X, all_train_y, scaler = prepare_multi_timestep_data(all_veh_data[:len(car_id)-2])
    # 分开预测
    test_X1, test_y1, _ = prepare_multi_timestep_data(all_veh_data[len(car_id)-2:len(car_id)-1], scaler=scaler)
    test_X2, test_y2, _ = prepare_multi_timestep_data(all_veh_data[len(car_id)-1:len(car_id)], scaler=scaler)


    batch_size = 64
    train_dataset = TimeSeriesDataset(all_train_X, all_train_y)
    test_dataset1 = TimeSeriesDataset(test_X1, test_y1)
    test_dataset2 = TimeSeriesDataset(test_X2, test_y2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True)
    # 测试集不打乱，就按照时间顺序预测，方便后续拼接
    test_loader1 = DataLoader(test_dataset1, batch_size=64, shuffle=False,num_workers=0, pin_memory=True)
    test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False,num_workers=0, pin_memory=True)


    # 创建模型实例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # 用于普通测试
    # model = Mamba_Agent(
    #     seq_len=100, in_channels=7, model_dim=64, d_state=64, agent_num=25,drop=0.2, pool='mean').to(device)

    # 用于联邦测试
    model = MambaAgentWithFedAWA(
        seq_len=100, in_channels=7, model_dim=64, d_state=64, drop=0.2, pool='mean').to(device)

    with torch.no_grad():
        dummy = torch.zeros(64, 100, 7, device=device)  # 按你的输入规格 _ =
        res=model(dummy)

    # # 用于普通测试
    state = torch.load(r"E:\PycharmProjects\my_battery_research\FedAWA_FNR\checkpoint\global_model_round_24.pth", map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # checkpoint = torch.load("目前最佳模型参数/op14.pth")
    # optimizer.load_state_dict(checkpoint)

    # train()
    # 保存模型权重
    # torch.save(model.state_dict(), "BAIC_MAF.pth")
    # torch.save(optimizer.state_dict(), "BAIC_MAF_op.pth")
    # print('完毕')

    # test(test_loader1, len(car_id)-2)
    test(test_loader2, len(car_id)-1)


