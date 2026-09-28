import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from client import Node, Client_update, Client_validate, validate
from server import Server_update
from fedawa import *   # 若需要导入 fedawa 中的辅助函数
import server

# 定义图像保存文件夹
FIG_DIR = './figures'
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------- 配置参数（直接定义）--------------------
class Config:
    # 数据参数
    window_size = 100
    forecast_horizon = 1
    data_path1 = r"E:\My_EV_set_0"
    data_path2 = r"E:\qinghua\battery_features"
    in_channels = 7        # 输入特征维度
    batchsize = 64   # 客户端本地训练的批次数
    validate_batchsize = 1  # 验证的批次数

    # 系统参数: 由于 2 个客户端的数据量不一样，所以尝试设置不一样的本地 epoch 训练
    device = '0'
    node_num = 2            # 固定两个客户端
    T = 8                   # 通信轮次（测试时可设小一点）
    extra_rounds = 22        # 额外训练的轮数，0表示使用原来的 T
    local_E = [5,10]        # 本地训练 epoch 数
    dataset = 'mydata'      # 仅用于标识，不影响加载
    select_ratio = 1.0      # 每轮参与聚合的客户端比例，1为全部参与
    local_model = 'Mamba_Agent'
    random_seed = 42   # 实验结果可重复显现
    exp_name = 'test_run'

    # 服务器参数（fedawa 专用）
    server_method = 'fedawa'
    # server_valid_ratio = 0.02   # 可能用不到，保留
    server_epochs = 10  # 这里先设置 10，不够再增加，AI建议是 5~20
    server_optimizer = 'adam'
    gamma = 1.0
    reg_distance = 'cos'

    # 客户端参数
    client_method = 'local_train'   # 可选 'local_train' 或 'fedprox'
    optimizer = 'adamw'
    # client_valid_ratio = 0.3        # 可能用不到
    lr = 0.001
    local_wd_rate = 1e-5
    momentum = 0.9
    mu = 0.001                       # FedProx 系数

args = Config()

# 设置 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = args.device
# 设置 GPU
if args.device.lower() != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # 可选，屏蔽 GPU
print('device:',device)

# -------------------- 检查点路径 --------------------
CHECKPOINT_DIR = './checkpoint'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'latest.pth')

# -------------------- 检查点保存与加载函数 --------------------
def save_checkpoint(round_num, central_node, global_T_weights,test_rmse_recorder,
                    test_mae_recorder, test_r2_recorder, test_mape_recorder,
                    avgtime, client_weights_history, args):
    CURRENT_ROUND_PATH = os.path.join(CHECKPOINT_DIR, f'global_model_round_{round_num+1}.pth')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'round': round_num,   # 保留训练轮数
        'model_state_dict': central_node.model.state_dict(),
        'global_T_weights': global_T_weights.cpu() if global_T_weights is not None else None,
        'test_rmse_recorder': test_rmse_recorder,
        'test_mae_recorder': test_mae_recorder,
        'test_r2_recorder': test_r2_recorder,
        'test_mape_recorder': test_mape_recorder,
        'avgtime': avgtime,
        'client_weights_history': client_weights_history,  # 新增
        'args': args.__dict__
    }
    torch.save(checkpoint, CURRENT_ROUND_PATH)
    print(f"Checkpoint saved at round {round_num+1}")

def load_checkpoint(central_node, args, device):
    if not os.path.exists(CHECKPOINT_PATH):  # 没有检查点的路径
        print("没有搜索到检查点的路径....")
        return 0, None, [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [[] for _ in range(4)], [],[]
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    central_node.model.load_state_dict(checkpoint['model_state_dict'])

    global_T_weights = checkpoint['global_T_weights']
    if global_T_weights is not None:
        global_T_weights = global_T_weights.to(device)

    start_round = checkpoint['round'] + 1  # 新开始的轮次数
    test_rmse_recorder = checkpoint['test_rmse_recorder']
    test_mae_recorder = checkpoint['test_mae_recorder']
    test_r2_recorder = checkpoint['test_r2_recorder']
    test_mape_recorder = checkpoint['test_mape_recorder']
    avgtime = checkpoint.get('avgtime', [])
    client_weights_history = checkpoint.get('client_weights_history', [])  # 新增

    # 恢复 args 中的参数（如学习率）
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        for key, value in saved_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
        print(f"Restored args from checkpoint, current lr = {args.lr}")
    else:
        print("Warning: No args found in checkpoint, using current args.")

    print('搜索到了检查点的路径，继续上次的训练开始新训练....')
    print(f"Loaded checkpoint from round {checkpoint['round']+1}")
    return start_round, global_T_weights, test_rmse_recorder, test_mae_recorder, test_r2_recorder, test_mape_recorder, avgtime, client_weights_history

# -------------------- 工具函数 --------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import random

class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)

def generate_selectlist(client_node, ratio=0.5):  # 生成选择的客户端
    candidate_list = list(range(len(client_node)))
    select_num = int(ratio * len(client_node))
    select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
    return select_list

def lr_scheduler(rounds, node_list, args):
    if rounds != 0:  # 从第二轮开始进行学习率调度
        args.lr *= 0.99
        for node in node_list:  # 遍历客户端的节点，每个节点都调整学习率
            node.args.lr = args.lr
            node.optimizer.param_groups[0]['lr'] = args.lr

# -------------------- 数据集定义 --------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_multi_timestep_data(all_veh_data, window_size=100, forecast_horizon=1, scaler=None):
    all_X, all_y = [], []
    if scaler is None:
        all_data = np.vstack(all_veh_data)
        max_abs_values = np.max(np.abs(all_data), axis=0)
        max_abs_values[max_abs_values == 0] = 1
        scaler = max_abs_values
    else:
        max_abs_values = scaler

    for car_data in all_veh_data:
        normalized_data = car_data / max_abs_values
        n_timesteps = normalized_data.shape[0]
        for i in range(0, n_timesteps - window_size - forecast_horizon + 1):
            X_window = normalized_data[i: i + window_size, 1:]   # 跳过第一列
            y_window = normalized_data[i + window_size: i + window_size + forecast_horizon, -1]
            all_X.append(X_window)
            all_y.append(y_window)

    return np.array(all_X), np.array(all_y), scaler

# -------------------- 数据加载 --------------------
def load_data(args):
    # 定义两个数据集的车辆编号
    car_id1 = list(range(1, 21))
    #car_id2 = [3,5,30,34,35,37,51,176,57,70,88,92,132,140,141,153,154,166,52,177]
    car_id2 = [3, 5, 15, 17, 24, 30, 34, 35, 37, 51, 57, 58, 70, 88,
              92, 109, 132, 140, 141, 153, 154, 166, 176, 52, 177]

    # 读取原始数据
    all_veh_data1, all_veh_data2 = [], []
    veh_ca1, veh_ca2 = [], []

    for i in range(len(car_id1)):
        id = car_id1[i]
        path = f"{args.data_path1}\#{id}.csv"
        veh = pd.read_csv(path)
        data = copy.deepcopy(veh.values)
        ca = copy.deepcopy(veh.values[:, -1])
        all_veh_data1.append(data)
        veh_ca1.append(list(ca))

    for i in range(len(car_id2)):
        id = car_id2[i]
        path = f"{args.data_path2}/#{id}.csv"
        veh = pd.read_csv(path)
        data = copy.deepcopy(veh.values)
        ca = copy.deepcopy(veh.values[:, -1])
        all_veh_data2.append(data)
        veh_ca2.append(list(ca))

    # 客户端 0
    train_data1 = all_veh_data1[:len(car_id1)-2]
    test_data1_1 = all_veh_data1[len(car_id1)-2:len(car_id1)-1]
    test_data1_2 = all_veh_data1[len(car_id1)-1:len(car_id1)]

    all_train_X1, all_train_y1, scaler1 = prepare_multi_timestep_data(
        train_data1, args.window_size, args.forecast_horizon
    )
    test_X1_1, test_y1_1, _ = prepare_multi_timestep_data(
        test_data1_1, args.window_size, args.forecast_horizon, scaler=scaler1
    )
    test_X1_2, test_y1_2, _ = prepare_multi_timestep_data(
        test_data1_2, args.window_size, args.forecast_horizon, scaler=scaler1
    )

    train_dataset1 = TimeSeriesDataset(all_train_X1, all_train_y1)
    test_dataset1_1 = TimeSeriesDataset(test_X1_1, test_y1_1)
    test_dataset1_2 = TimeSeriesDataset(test_X1_2, test_y1_2)

    train_loader0 = DataLoader(train_dataset1, batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True)
    # 测试集
    # test_dataset1 = torch.utils.data.ConcatDataset([test_dataset1_1, test_dataset1_2])
    # 客户端的验证集就是测试集
    client_val_loader1_1 = DataLoader(test_dataset1_1, batch_size=args.validate_batchsize,
                                      shuffle=False, num_workers=0, pin_memory=True)
    client_val_loader1_2 = DataLoader(test_dataset1_2, batch_size=args.validate_batchsize,
                                      shuffle=False, num_workers=0, pin_memory=True)

    # 客户端1
    train_data2 = all_veh_data2[:len(car_id2)-2]
    test_data2_1 = all_veh_data2[len(car_id2)-2:len(car_id2)-1]
    test_data2_2 = all_veh_data2[len(car_id2)-1:len(car_id2)]

    all_train_X2, all_train_y2, scaler2 = prepare_multi_timestep_data(
        train_data2, args.window_size, args.forecast_horizon
    )
    test_X2_1, test_y2_1, _ = prepare_multi_timestep_data(
        test_data2_1, args.window_size, args.forecast_horizon, scaler=scaler2
    )
    test_X2_2, test_y2_2, _ = prepare_multi_timestep_data(
        test_data2_2, args.window_size, args.forecast_horizon, scaler=scaler2
    )

    train_dataset2 = TimeSeriesDataset(all_train_X2, all_train_y2)
    test_dataset2_1 = TimeSeriesDataset(test_X2_1, test_y2_1)
    test_dataset2_2 = TimeSeriesDataset(test_X2_2, test_y2_2)

    train_loader1 = DataLoader(train_dataset2, batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=True)
    # 测试集
    # test_dataset2 = torch.utils.data.ConcatDataset([test_dataset2_1, test_dataset2_2])
    # 客户端的验证集就是测试集
    client_val_loader2_1 = DataLoader(test_dataset2_1, batch_size=args.validate_batchsize,
                                      shuffle=False, num_workers=0, pin_memory=True)
    client_val_loader2_2 = DataLoader(test_dataset2_2, batch_size=args.validate_batchsize,
                                      shuffle=False, num_workers=0, pin_memory=True)

    # 全局测试集，客户端的2个测试集的集合
    '''
    global_test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])
    test_loader = DataLoader(global_test_dataset, batch_size=args.validate_batchsize, shuffle=False, num_workers=0, pin_memory=True)
    '''
    # 组织返回
    train_loaders = [train_loader0, train_loader1]
    # 这里设置 val_loaders = test_loader
    val_loaders = [[client_val_loader1_1,client_val_loader1_2],[client_val_loader2_1,client_val_loader2_2]]
    test_loader=[[client_val_loader1_1,client_val_loader1_2],[client_val_loader2_1,client_val_loader2_2]]
    scalers = [scaler1, scaler2]
    veh_cas = [veh_ca1, veh_ca2]
    car_ids = [car_id1, car_id2]

    return train_loaders, val_loaders, test_loader, scalers, veh_cas, car_ids

# -------------------- 全局模型最终测试函数 --------------------
def final_test_for_global(args, node, test_loader, scaler, veh_ca, car_id, indx,client_name,round_num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    capacity_scaler = scaler[-1]
    mae_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    test_mae, test_mse, test_rmse = [], [], []
    predict, real = [], []

    with torch.no_grad():
        node.model.eval()
        for features, capacity in test_loader:
            features, capacity = features.to(device), capacity.to(device)
            outputs = node.model(features)

            test_mse.append(mse_loss(outputs, capacity).item())
            test_mae.append(mae_loss(outputs, capacity).item())
            test_rmse.append(torch.sqrt(mse_loss(outputs, capacity)).item())

            outputs_np = outputs.cpu().numpy()
            capacity_np = capacity.cpu().numpy()
            outputs_restored = outputs_np * capacity_scaler
            capacity_restored = capacity_np * capacity_scaler
            predict.extend(outputs_restored.flatten().tolist())
            real.extend(capacity_restored.flatten().tolist())

    mean_mse = np.mean(test_mse)
    mean_mae = np.mean(test_mae)
    mean_rmse = np.mean(test_rmse)
    mean_r2 = r2_score(real, predict)
    mean_mape = np.mean(np.abs((np.array(real) - np.array(predict)) / (np.array(real) + 1e-8))) * 100

    # 总体图（简化）
    fig1 = plt.figure(figsize=(16, 10))
    total_ca = veh_ca[indx]  # 仅示例，实际需对应测试车辆
    # plt.figure(figsize=(16, 20))
    plt.plot(total_ca, 'ro-', markersize=2, label='Real')
    plt.plot(np.arange(100, 100+len(predict)), predict, 'bo-', markersize=2, label='Model')
    plt.axvline(x=100, c='k', ls='--')
    plt.title(f'Vehicle {car_id[indx]}')
    plt.ylabel('Capacity (Ah)')
    plt.xlabel('Cycle')
    # plt.show()
    save_path1 = os.path.join(FIG_DIR, f'{client_name}_vehicle{car_id[indx]}_round{round_num}.png')
    plt.savefig(save_path1)
    plt.close(fig1)
    result = {'mse': mean_mse, 'mae': mean_mae, 'rmse': mean_rmse, 'r2': mean_r2,'mape':mean_mape}
    return result

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    setup_seed(args.random_seed)
    print('Config:', args.__dict__)

    # 加载数据
    train_loaders, val_loaders, test_loader, scalers, veh_cas, car_ids = load_data(args)

    # 聚合权重
    # sample_size = [len(loader.dataset) for loader in train_loaders]
    # size_weights = [s / sum(sample_size) for s in sample_size]
    size_weights=[0.5, 0.5]
    print('Aggregation weights:', size_weights)

    # 创建中央节点（服务器）: 中心服务器的本地训练集和本地验证集都是测试集
    central_node = Node(-1, test_loader, test_loader, args, scaler=scalers)
    # 创建客户端节点,初始是用同一套参数 args
    client_nodes = []
    client_nodes.append(Node(0, train_loaders[0], val_loaders[0], args, scaler=scalers[0]))
    client_nodes.append(Node(1, train_loaders[1], val_loaders[1], args, scaler=scalers[1]))

    # 记录器
    # final_test_recorder = RunningAverage()

    # 尝试加载检查点
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_round, loaded_T_weights, test_rmse_recorder, test_mae_recorder, test_r2_recorder, test_mape_recorder, avgtime, client_weights_history = load_checkpoint(
        central_node, args, device)

    # 如果从头开始，初始化历史记录为空列表
    if start_round == 0:
        client_weights_history = []

    # 如果设置了 extra_rounds，则调整总轮数 T
    if args.extra_rounds > 0:
        args.T = start_round + args.extra_rounds
        print(f"Resuming with extra rounds: total rounds set to {args.T}")

    # 在加载检查点后，将可能为列表的权重转为张量
    if loaded_T_weights is not None:
        if isinstance(loaded_T_weights, list):
            loaded_T_weights = torch.tensor(loaded_T_weights, dtype=torch.float32).to(device)
        server.global_T_weights = loaded_T_weights
    else:
        server.global_T_weights = None

    # 初始化记录器（如果未加载则新建，但加载函数已返回正确的列表）
    # 这里只需确保变量存在
    if start_round == 0:
        # 如果从头开始，初始化记录器（已在加载函数中返回空列表，但为了清晰，可再次初始化）
        # 记录全局测试时每个测试集的指标变化
        test_rmse_recorder = [[] for _ in range(4)]
        test_mae_recorder = [[] for _ in range(4)]
        test_r2_recorder = [[] for _ in range(4)]
        test_mape_recorder = [[] for _ in range(4)]
        avgtime = []

    for rounds in range(start_round, args.T):  # 开始进行多轮通信
        print(f'\n=============== 开始 Round {rounds+1} 通信 ===============')

        # 学习率调整（可选）
        lr_scheduler(rounds, client_nodes, args)

        # 客户端更新
        print('================= 开始进行客户端的训练更新 ================')
        client_nodes, t_client_mse,t_client_mae, t_client_rmse = Client_update(args, client_nodes, central_node)
        print('客户端的综合训练结果如下所示：')
        print(f'Client training MSE:',t_client_mse)
        print(f'Client training MAE:', t_client_mae)
        print(f'Client training RMSE:', t_client_rmse)

        # 客户端验证
        print('================= 开始进行客户端训练后验证 ================')
        v_client_mse, v_client_mae, v_client_rmse, v_client_r2, v_client_mape = Client_validate(args, client_nodes)
        print('客户端验证结果如下所示:')
        print(f'Client validating MSE:',v_client_mse)
        print(f'Client validating MAE:', v_client_mae)
        print(f'Client validating RMSE:', v_client_rmse)
        print(f'Client validating R2:', v_client_r2)
        print(f'Client validating MAPE:', v_client_mape)


        # 选择客户端
        select_list = list(range(len(client_nodes))) if args.select_ratio == 1.0 else generate_selectlist(client_nodes, args.select_ratio)

        # 服务器聚合
        print('================= 开始进行服务器聚合 ================')
        start = time.perf_counter()
        central_node, prob = Server_update(args, central_node, client_nodes, select_list,
                                           size_weights, rounds_num=rounds)
        end = time.perf_counter()
        print(f'Server update time: {end - start:.4f}s')
        avgtime.append(end - start)

        # 记录当前客户端的聚合权重概率（转换为列表以便存储）
        client_weights_history.append(prob.cpu().tolist())

        # 全局模型测试
        print('================= 开始进行服务器聚合后的全局测试 ================')
        print('================= 开始进行客户端1第一个测试集的测试 ================')
        result1=final_test_for_global(args, central_node, val_loaders[0][0], scalers[0], veh_cas[0],
                                      car_ids[0], 18,'Client1',rounds+1)
        print('客户端1第一个测试集的结果：',result1)

        print('================= 开始进行客户端1第二个测试集的测试 ================')
        result2=final_test_for_global(args, central_node, val_loaders[0][1], scalers[0], veh_cas[0],
                                      car_ids[0], 19,'Client1',rounds+1)
        print('客户端1第二个测试集的结果：', result2)

        print('================= 开始进行客户端2第一个测试集的测试 ================')
        result3=final_test_for_global(args, central_node, val_loaders[1][0], scalers[1], veh_cas[1],
                                      car_ids[1], 23,'Client2',rounds+1)
        print('客户端2第一个测试集的结果：', result3)

        print('================= 开始进行客户端2第二个测试集的测试 ================')
        result4=final_test_for_global(args, central_node, val_loaders[1][1], scalers[1], veh_cas[1],
                                      car_ids[1], 24,'Client2',rounds+1)
        print('客户端2第二个测试集的结果：', result4)

        total_results=[result1,result2,result3,result4]
        for i in range(4):
            test_rmse_recorder[i].append(total_results[i]['rmse'])
            test_mae_recorder[i].append(total_results[i]['mae'])
            test_r2_recorder[i].append(total_results[i]['r2'])
            test_mape_recorder[i].append(total_results[i]['mape'])

        # 在每轮末尾，所有记录更新后保存检查点
        save_checkpoint(rounds, central_node, server.global_T_weights,test_rmse_recorder,
                        test_mae_recorder, test_r2_recorder, test_mape_recorder,
                        avgtime, client_weights_history,args)

    print(f'\nAverage server update time: {np.mean(avgtime):.4f}s')

    print('\n=============== 所有通信轮数下全局模型测试结果的变化 ===============')
    # 指标 1 的变化
    print('\n=============== 每个测试集 RMSE 的变化 ===============')
    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_rmse_recorder[i], label=f'Test{i + 1}')
    plt.title('RMSE Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'RMSE_global_change.png'))
    plt.close()

    # 指标 2 的变化
    print('\n=============== 每个测试集 MAE 的变化 ===============')
    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_mae_recorder[i], label=f'Test{i + 1}')
    plt.title('MAE Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'MAE_global_change.png'))
    plt.close()

    # 指标 3 的变化
    print('\n=============== 每个测试集 R2 的变化 ===============')
    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_r2_recorder[i], label=f'Test{i + 1}')
    plt.title('R2 Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('R2')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'R2_global_change.png'))
    plt.close()


    # 指标 4 的变化
    print('\n=============== 每个测试集 MAPE 的变化 ===============')
    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_mape_recorder[i], label=f'Test{i + 1}')
    plt.title('MAPE Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('MAPE')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'MAPE_global_change.png'))
    plt.close()

    print('Experiment finished.')

    # 在所有指标变化图之后, 绘制每次聚合时权重变化曲线
    print('\n=============== 绘制每次聚合时权重变化曲线 ===============')
    if len(client_weights_history) > 0:
        weights_array = np.array(client_weights_history)  # shape: (num_rounds, num_clients)
        plt.figure(figsize=(16, 10))
        for i in range(weights_array.shape[1]):
            plt.plot(weights_array[:, i], label=f'Client {i} weight')
        plt.xlabel('Rounds')
        plt.ylabel('Aggregation Weight')
        plt.title('Client Aggregation Weights over Rounds')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(FIG_DIR, 'client_weights.png'))
        plt.close()

    print('Experiment finished.')

