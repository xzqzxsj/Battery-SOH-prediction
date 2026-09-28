import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from client_1 import Node, Client_update, Client_validate
from server import Server_update
from fedawa import *
import server
import random

FIG_DIR = r"E:\PycharmProjects\my_battery_research\论文实验\figures"
os.makedirs(FIG_DIR, exist_ok=True)

CHECKPOINT_DIR = r"E:\PycharmProjects\my_battery_research\论文实验\checkpoint"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'latest.pth')


# -------------------- 配置参数 --------------------
class Config:
    # 数据参数
    window_size = 100
    forecast_horizon = 1
    data_path1 = r"E:\My_EV_set_0\SOH_data"
    data_path2 = r"E:\qinghua\SOH_data"
    in_channels = 7
    batchsize = 64
    validate_batchsize = 64

    # 系统参数
    device = '0'
    node_num = 2
    T = 40
    extra_rounds = 0   # 在最后的代码处设置，防止被掩盖掉
    local_E = [5, 5]
    dataset = 'mydata'
    select_ratio = 1.0
    local_model = 'Mamba_Agent'
    random_seed = 42
    exp_name = 'test_run'

    # 服务器参数
    # server_method = 'fedawa'
    server_method = 'fedavg'
    server_epochs = 10
    server_optimizer = 'adam'
    gamma = 1.0
    reg_distance = 'cos'

    # 客户端参数
    # client_method = 'feddyn'   # 'local_train' / 'fedprox' / 'feddyn'
    client_method = 'local_train'
    optimizer = 'adamw'
    lr = 0.001
    local_wd_rate = 1e-5
    momentum = 0.9
    mu = 0.001
    feddyn_alpha = [1e-4, 1e-4]


args = Config()


# -------------------- 配置与随机状态工具 --------------------
def config_to_dict(cfg):
    result = {}
    for key in dir(cfg):
        if key.startswith('_'):
            continue
        value = getattr(cfg, key)
        if callable(value):
            continue
        result[key] = copy.deepcopy(value)
    return result


def update_config_from_dict(cfg, cfg_dict, skip_keys=None):
    if skip_keys is None:
        skip_keys = set()
    for key, value in cfg_dict.items():
        if key in skip_keys:
            continue
        setattr(cfg, key, value)
    return cfg


# def get_rng_state():
#     state = {
#         'python_random_state': random.getstate(),
#         'numpy_random_state': np.random.get_state(),
#         'torch_random_state': torch.get_rng_state(),
#     }
#     if torch.cuda.is_available():
#         state['torch_cuda_random_state_all'] = torch.cuda.get_rng_state_all()
#     else:
#         state['torch_cuda_random_state_all'] = None
#     return state


# def set_rng_state(state_dict):
#     if not state_dict:
#         return
#     if 'python_random_state' in state_dict and state_dict['python_random_state'] is not None:
#         random.setstate(state_dict['python_random_state'])
#     if 'numpy_random_state' in state_dict and state_dict['numpy_random_state'] is not None:
#         np.random.set_state(state_dict['numpy_random_state'])
#     if 'torch_random_state' in state_dict and state_dict['torch_random_state'] is not None:
#         torch.set_rng_state(state_dict['torch_random_state'])
#     if torch.cuda.is_available():
#         cuda_state = state_dict.get('torch_cuda_random_state_all', None)
#         if cuda_state is not None:
#             torch.cuda.set_rng_state_all(cuda_state)


def restore_args_from_latest_if_exists(cfg):
    if not os.path.exists(CHECKPOINT_PATH):
        return cfg
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    saved_args = checkpoint.get('args', None)
    if saved_args is not None:
        update_config_from_dict(cfg, saved_args)
        print('Loaded args from latest checkpoint before initialization.')

    return cfg


# 在创建 node 前先恢复配置，避免优化器/训练模式初始化错误
args = restore_args_from_latest_if_exists(args)

# 设置 GPU
if str(args.device).lower() != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

print('device:', device)


# -------------------- 检查点保存与加载 --------------------
def save_checkpoint(round_num, central_node, global_T_weights, test_rmse_recorder,
                    test_mae_recorder, test_r2_recorder, test_mape_recorder,
                    avgtime, client_weights_history, args, client_nodes=None, size_weights=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    current_round_path = os.path.join(CHECKPOINT_DIR, f'global_model_round_{round_num+1}.pth')

    checkpoint = {
        'round': round_num,
        'model_state_dict': central_node.model.state_dict(),
        'global_T_weights': global_T_weights.cpu() if global_T_weights is not None else None,
        'test_rmse_recorder': test_rmse_recorder,
        'test_mae_recorder': test_mae_recorder,
        'test_r2_recorder': test_r2_recorder,
        'test_mape_recorder': test_mape_recorder,
        'avgtime': avgtime,
        'client_weights_history': client_weights_history,
        'args': config_to_dict(args),
        'size_weights': copy.deepcopy(size_weights),
        # 'rng_state': get_rng_state(),
    }

    if client_nodes is not None:
        checkpoint['client_optimizer_states'] = [
            copy.deepcopy(node.optimizer.state_dict()) for node in client_nodes
        ]
        checkpoint['client_feddyn_states'] = [
            node.feddyn_y.detach().cpu() if getattr(node, 'feddyn_y', None) is not None else None
            for node in client_nodes
        ]

    torch.save(checkpoint, current_round_path)
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at round {round_num+1}")
    print(f"Latest checkpoint updated: {CHECKPOINT_PATH}")


def load_checkpoint(central_node, client_nodes, args, device):
    if not os.path.exists(CHECKPOINT_PATH):
        print("没有搜索到检查点路径，开始全新训练。")
        return {
            'start_round': 0,
            'global_T_weights': None,
            'test_rmse_recorder': [[] for _ in range(4)],
            'test_mae_recorder': [[] for _ in range(4)],
            'test_r2_recorder': [[] for _ in range(4)],
            'test_mape_recorder': [[] for _ in range(4)],
            'avgtime': [],
            'client_weights_history': [],
            'size_weights': None,
        }

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # 恢复配置
    saved_args = checkpoint.get('args', None)
    if saved_args is not None:
        update_config_from_dict(args, saved_args)
        print(f"Restored args from checkpoint, current lr = {args.lr}")

    # 恢复中央模型
    central_node.model.load_state_dict(checkpoint['model_state_dict'])

    # 恢复全局聚合权重
    global_T_weights = checkpoint.get('global_T_weights', None)
    if global_T_weights is not None:
        global_T_weights = global_T_weights.to(device)

    # 恢复客户端状态
    client_feddyn_states = checkpoint.get('client_feddyn_states', None)
    if client_feddyn_states is not None:
        for idx, state in enumerate(client_feddyn_states):
            if idx < len(client_nodes) and state is not None:
                client_nodes[idx].feddyn_y = state.to(device)

    client_optimizer_states = checkpoint.get('client_optimizer_states', None)
    if client_optimizer_states is not None:
        for idx, opt_state in enumerate(client_optimizer_states):
            if idx < len(client_nodes) and opt_state is not None:
                try:
                    client_nodes[idx].optimizer.load_state_dict(opt_state)
                except Exception as e:
                    print(f"Warning: failed to load optimizer state for client {idx}: {e}")

    # 恢复 RNG
    # set_rng_state(checkpoint.get('rng_state', None))
    # try:
    #     set_rng_state(checkpoint.get('rng_state', None))
    # except Exception as e:
    #     print(f"Warning: skip restoring rng_state due to error: {e}")

    print(f"Loaded checkpoint from round {checkpoint['round'] + 1}")
    print('搜索到了检查点路径，继续上次训练。')

    return {
        'start_round': checkpoint['round'] + 1,
        'global_T_weights': global_T_weights,
        'test_rmse_recorder': checkpoint.get('test_rmse_recorder', [[] for _ in range(4)]),
        'test_mae_recorder': checkpoint.get('test_mae_recorder', [[] for _ in range(4)]),
        'test_r2_recorder': checkpoint.get('test_r2_recorder', [[] for _ in range(4)]),
        'test_mape_recorder': checkpoint.get('test_mape_recorder', [[] for _ in range(4)]),
        'avgtime': checkpoint.get('avgtime', []),
        'client_weights_history': checkpoint.get('client_weights_history', []),
        'size_weights': checkpoint.get('size_weights', None),
    }


# -------------------- 工具函数 --------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


def generate_selectlist(client_node, ratio=0.5):
    candidate_list = list(range(len(client_node)))
    select_num = int(ratio * len(client_node))
    select_list = np.random.choice(candidate_list, select_num, replace=False).tolist()
    return select_list


def lr_scheduler(rounds, node_list, args):
    if rounds != 0:
        args.lr *= 0.95  # 从 0.99 改成 0.95
        for node in node_list:
            node.args.lr = args.lr
            for group in node.optimizer.param_groups:
                group['lr'] = args.lr


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
            X_window = normalized_data[i: i + window_size, 1:]
            y_window = normalized_data[i + window_size: i + window_size + forecast_horizon, -1]
            all_X.append(X_window)
            all_y.append(y_window)

    return np.array(all_X), np.array(all_y), scaler


# -------------------- 数据加载 --------------------
def load_data(args):
    car_id1 = list(range(1, 21))
    car_id2 = [3, 52, 15, 17, 24, 30, 34, 35, 37, 51, 57, 58, 70, 88,
               92, 109, 132, 140, 141, 153, 154, 166, 176, 5, 177]

    all_veh_data1, all_veh_data2 = [], []
    veh_ca1, veh_ca2 = [], []

    for i in range(len(car_id1)):
        cid = car_id1[i]
        path = f"{args.data_path1}\\#{cid}.csv"
        veh = pd.read_csv(path)
        data = copy.deepcopy(veh.values)
        ca = copy.deepcopy(veh.values[:, -1])
        all_veh_data1.append(data)
        veh_ca1.append(list(ca))

    for i in range(len(car_id2)):
        cid = car_id2[i]
        path = f"{args.data_path2}/#{cid}.csv"
        veh = pd.read_csv(path)
        data = copy.deepcopy(veh.values)
        ca = copy.deepcopy(veh.values[:, -1])
        all_veh_data2.append(data)
        veh_ca2.append(list(ca))

    # 客户端0
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
    client_val_loader1_1 = DataLoader(test_dataset1_1, batch_size=args.validate_batchsize, shuffle=False, num_workers=0, pin_memory=True)
    client_val_loader1_2 = DataLoader(test_dataset1_2, batch_size=args.validate_batchsize, shuffle=False, num_workers=0, pin_memory=True)

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
    client_val_loader2_1 = DataLoader(test_dataset2_1, batch_size=args.validate_batchsize, shuffle=False, num_workers=0, pin_memory=True)
    client_val_loader2_2 = DataLoader(test_dataset2_2, batch_size=args.validate_batchsize, shuffle=False, num_workers=0, pin_memory=True)

    train_loaders = [train_loader0, train_loader1]
    val_loaders = [[client_val_loader1_1, client_val_loader1_2], [client_val_loader2_1, client_val_loader2_2]]
    test_loader = [[client_val_loader1_1, client_val_loader1_2], [client_val_loader2_1, client_val_loader2_2]]
    scalers = [scaler1, scaler2]
    veh_cas = [veh_ca1, veh_ca2]
    car_ids = [car_id1, car_id2]

    return train_loaders, val_loaders, test_loader, scalers, veh_cas, car_ids


# -------------------- 全局模型最终测试 --------------------
def final_test_for_global(args, node, test_loader, scaler, veh_ca, car_id, indx, client_name, round_num):
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

    fig1 = plt.figure(figsize=(16, 10))
    total_ca = veh_ca[indx]
    plt.plot(total_ca, 'ro-', markersize=2, label='Real')
    plt.plot(np.arange(100, 100 + len(predict)), predict, 'bo-', markersize=2, label='Model')
    plt.axvline(x=100, c='k', ls='--')
    plt.title(f'Vehicle {car_id[indx]}')
    plt.ylabel('SOH')
    plt.xlabel('Cycle')
    save_path1 = os.path.join(FIG_DIR, f'{client_name}_vehicle{car_id[indx]}_round{round_num}.png')
    plt.savefig(save_path1)
    plt.close(fig1)

    result = {'mse': mean_mse, 'mae': mean_mae, 'rmse': mean_rmse, 'r2': mean_r2, 'mape': mean_mape}
    return result


# -------------------- 主程序 --------------------
if __name__ == '__main__':
    setup_seed(args.random_seed)
    print('Config:', config_to_dict(args))

    train_loaders, val_loaders, test_loader, scalers, veh_cas, car_ids = load_data(args)

    # sample_size = [len(loader.dataset) for loader in train_loaders]
    # # 初始权重按照数据集大小划分
    # initial_size_weights = [s / sum(sample_size) for s in sample_size]
    initial_size_weights=[0.5, 0.5]
    print('Initial aggregation weights:', initial_size_weights)

    central_node = Node(-1, test_loader, test_loader, args, scaler=scalers)

    client_nodes = [
        Node(0, train_loaders[0], val_loaders[0], args, scaler=scalers[0]),
        Node(1, train_loaders[1], val_loaders[1], args, scaler=scalers[1]),
    ]

    ckpt = load_checkpoint(central_node, client_nodes, args, device)

    start_round = ckpt['start_round']
    test_rmse_recorder = ckpt['test_rmse_recorder']
    test_mae_recorder = ckpt['test_mae_recorder']
    test_r2_recorder = ckpt['test_r2_recorder']
    test_mape_recorder = ckpt['test_mape_recorder']
    avgtime = ckpt['avgtime']
    client_weights_history = ckpt['client_weights_history']
    size_weights = ckpt['size_weights'] if ckpt['size_weights'] is not None else initial_size_weights

    # args.extra_rounds=10

    if args.extra_rounds > 0:
        args.T = start_round + args.extra_rounds
        print(f"Resuming with extra rounds: total rounds set to {args.T}")

    if ckpt['global_T_weights'] is not None:
        server.global_T_weights = ckpt['global_T_weights']
    else:
        server.global_T_weights = None

    for rounds in range(start_round, args.T):
        print(f'\n=============== 开始 Round {rounds+1} 通信 ===============')

        lr_scheduler(rounds, client_nodes, args)

        print('================= 开始进行客户端的训练更新 ================')
        client_nodes, t_client_mse, t_client_mae, t_client_rmse = Client_update(args, client_nodes, central_node)
        print('客户端的综合训练结果如下所示：')
        print(f'Client training MSE: {t_client_mse}')
        print(f'Client training MAE: {t_client_mae}')
        print(f'Client training RMSE: {t_client_rmse}')

        print('================= 开始进行客户端训练后验证 ================')
        v_client_mse, v_client_mae, v_client_rmse, v_client_r2, v_client_mape = Client_validate(args, client_nodes)
        print('客户端验证结果如下所示:')
        print(f'Client validating MSE: {v_client_mse}')
        print(f'Client validating MAE: {v_client_mae}')
        print(f'Client validating RMSE: {v_client_rmse}')
        print(f'Client validating R2: {v_client_r2}')
        print(f'Client validating MAPE: {v_client_mape}')

        if args.select_ratio == 1.0:
            select_list = list(range(len(client_nodes)))
        else:
            select_list = generate_selectlist(client_nodes, args.select_ratio)

        print('================= 开始进行服务器聚合 ================')
        start = time.perf_counter()
        central_node, prob = Server_update(args, central_node, client_nodes, select_list, size_weights, rounds_num=rounds)
        end = time.perf_counter()
        print(f'Server update time: {end - start:.4f}s')
        avgtime.append(end - start)

        client_weights_history.append(prob.cpu().tolist())

        print('================= 开始进行服务器聚合后的全局测试 ================')
        result1 = final_test_for_global(args, central_node, val_loaders[0][0], scalers[0], veh_cas[0], car_ids[0], 18, 'Client1', rounds+1)
        print('客户端1第一个测试集的结果：', result1)

        result2 = final_test_for_global(args, central_node, val_loaders[0][1], scalers[0], veh_cas[0], car_ids[0], 19, 'Client1', rounds+1)
        print('客户端1第二个测试集的结果：', result2)

        result3 = final_test_for_global(args, central_node, val_loaders[1][0], scalers[1], veh_cas[1], car_ids[1], 23, 'Client2', rounds+1)
        print('客户端2第一个测试集的结果：', result3)

        result4 = final_test_for_global(args, central_node, val_loaders[1][1], scalers[1], veh_cas[1], car_ids[1], 24, 'Client2', rounds+1)
        print('客户端2第二个测试集的结果：', result4)

        total_results = [result1, result2, result3, result4]
        for i in range(4):
            test_rmse_recorder[i].append(total_results[i]['rmse'])
            test_mae_recorder[i].append(total_results[i]['mae'])
            test_r2_recorder[i].append(total_results[i]['r2'])
            test_mape_recorder[i].append(total_results[i]['mape'])

        save_checkpoint(
            rounds,
            central_node,
            server.global_T_weights,
            test_rmse_recorder,
            test_mae_recorder,
            test_r2_recorder,
            test_mape_recorder,
            avgtime,
            client_weights_history,
            args,
            client_nodes=client_nodes,
            size_weights=size_weights
        )

    print(f'\nAverage server update time: {np.mean(avgtime):.4f}s')

    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_rmse_recorder[i], label=f'Test{i + 1}')
    plt.title('RMSE Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'RMSE_global_change.png'))
    plt.close()

    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_mae_recorder[i], label=f'Test{i + 1}')
    plt.title('MAE Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'MAE_global_change.png'))
    plt.close()

    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_r2_recorder[i], label=f'Test{i + 1}')
    plt.title('R2 Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('R2')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'R2_global_change.png'))
    plt.close()

    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.plot(test_mape_recorder[i], label=f'Test{i + 1}')
    plt.title('MAPE Global Change')
    plt.xlabel('Rounds')
    plt.ylabel('MAPE')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, 'MAPE_global_change.png'))
    plt.close()

    if len(client_weights_history) > 0:
        weights_array = np.array(client_weights_history)
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