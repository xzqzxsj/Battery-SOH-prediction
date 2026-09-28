# 训练带有 UQ 的模型
import os
import copy
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from MAgent.Mamba_Agent import Mamba_Agent
from MAgent.MAgent_AWA import MambaAgentWithFedAWA
from uq_utils import absolute_residual_score, run_uq_method, compute_uq_metrics

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mape_func(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 数据准备
def compute_scaler(all_veh_data: List[np.ndarray]) -> np.ndarray:
    all_data = np.vstack(all_veh_data)
    max_abs_values = np.max(np.abs(all_data), axis=0)
    max_abs_values[max_abs_values == 0] = 1.0
    return max_abs_values


def prepare_multi_timestep_data(all_veh_data,window_size=100,forecast_horizon=1,scaler=None,):
    all_X, all_y = [], []
    if scaler is None:
        scaler = compute_scaler(all_veh_data)
    for car_data in all_veh_data:
        normalized_data = car_data / scaler
        n_timesteps = normalized_data.shape[0]
        for i in range(0, n_timesteps - window_size - forecast_horizon + 1):
            X_window = normalized_data[i:i + window_size, 1:]  # [window, 7]
            y_window = normalized_data[i + window_size:i + window_size + forecast_horizon, -1]
            all_X.append(X_window)
            all_y.append(y_window)
    return np.array(all_X), np.array(all_y), scaler


def prepare_single_vehicle_test_data(veh_data,scaler,window_size=100,forecast_horizon= 1):
    X, y, _ = prepare_multi_timestep_data([veh_data], window_size, forecast_horizon, scaler)
    return X, y

# 单轮训练
def run_epoch(model, loader, optimizer, device):
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    model.train()  # 训练
    total_mae, total_mse, total_rmse = 0.0, 0.0, 0.0

    for features, capacity in loader:
        features = features.to(device)
        capacity = capacity.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = mse_loss(outputs, capacity)
        mae = mae_loss(outputs, capacity)
        rmse = torch.sqrt(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_mae += mae.item()
        total_mse += loss.item()
        total_rmse += rmse.item()

    n = len(loader)
    return {"mae": total_mae / n, "mse": total_mse / n, "rmse": total_rmse / n}


@torch.no_grad()
def evaluate_point_loader(model, loader, device):
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    model.eval()  #  测试
    total_mae, total_mse, total_rmse = 0.0, 0.0, 0.0

    for features, capacity in loader:
        features = features.to(device)
        capacity = capacity.to(device)
        outputs = model(features)

        mse = mse_loss(outputs, capacity)
        mae = mae_loss(outputs, capacity)
        rmse = torch.sqrt(mse)

        total_mae += mae.item()
        total_mse += mse.item()
        total_rmse += rmse.item()

    n = len(loader)
    return {"mae": total_mae / n, "mse": total_mse / n, "rmse": total_rmse / n}


# 收集反归一化后的预测值
@torch.no_grad()
def collect_predictions(model, loader, device, capacity_scaler):
    model.eval()
    preds, trues = [], []  # 记录反归一化后的结果
    stand_preds, stand_trues=[],[]  # 记录归一化后的结果
    for features, capacity in loader:
        features = features.to(device)
        outputs = model(features)

        outputs_np = outputs.cpu().numpy().reshape(-1)
        capacity_np = capacity.cpu().numpy().reshape(-1)

        stand_preds.extend(outputs_np.tolist())
        stand_trues.extend(capacity_np.tolist())

        # 反归一化
        preds.extend((outputs_np * capacity_scaler).tolist())
        trues.extend((capacity_np * capacity_scaler).tolist())
    return np.array(preds), np.array(trues), np.array(stand_preds), np.array(stand_trues)



def train_model(model,train_loader,optimizer,device,epochs):
    print("Starting training ...")
    for epo in range(epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        print(
            f"Epoch {epo+1} | "
            f"Train MAE={train_metrics['mae']:.6f}, MSE={train_metrics['mse']:.6f}, RMSE={train_metrics['rmse']:.6f}")
    return model


# UQ 测试
@torch.no_grad()
def collect_calibration_scores(model, val_loader, device, capacity_scaler):
    y_pred, y_true, stand_y_pred, stand_y_true = collect_predictions(model, val_loader, device, capacity_scaler)
    scores = absolute_residual_score(y_true, y_pred)
    return scores


@torch.no_grad()
def predict_test_vehicle_with_uq(
    model,
    test_loader,
    warmup_scores,
    scaler,
    raw_capacity_curve,
    vehicle_id,
    device,
    alpha=0.1,
    uq_method="Quantile+Integrator(log)",
    uq_lr=0.01,
    T_burnin=30,
    uq_method_kwargs=None,
    plot_dir="plots_uq",
):
    if uq_method_kwargs is None:
        uq_method_kwargs = {
            "Csat": 3.0,
            "KI": 1.0,
            "proportional_lr": True,
        }

    capacity_scaler = float(scaler[-1])
    y_pred, y_true, stand_y_pred, stand_y_true = collect_predictions(model, test_loader, device, capacity_scaler)

    test_scores = absolute_residual_score(y_true, y_pred)
    # 把验证集用于预热的残差和测试集的残差拼起来
    scores_for_uq = np.concatenate([warmup_scores, test_scores], axis=0)
    prefix_len = len(warmup_scores)

    uq_result = run_uq_method(
        scores=scores_for_uq,
        method_name=uq_method,
        alpha=alpha,
        lr=uq_lr,
        ahead=1,
        T_burnin=T_burnin,
        method_kwargs=uq_method_kwargs,
    )

    q_all = np.asarray(uq_result["q"])
    q_test = np.maximum(q_all[prefix_len:], 0.0)

    lower = y_pred - q_test
    upper = y_pred + q_test

    mse = np.mean((stand_y_pred - stand_y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(stand_y_pred - stand_y_true))
    mape = mape_func(stand_y_true, stand_y_pred)
    r2 = r2_score(stand_y_true, stand_y_pred)

    uq_metrics = compute_uq_metrics(
        y_true=y_true,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        q=q_test,
        alpha=alpha,
    )

    # 计算各种测试指标
    result = {
        "point_metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
        },
        "uq_metrics": uq_metrics,
    }

    plot_capacity_with_uq(
        raw_capacity_curve=np.asarray(raw_capacity_curve, dtype=float),
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        vehicle_id=vehicle_id,
        window_size=100,
        save_dir=plot_dir,
    )

    return result



# 可视化
def plot_capacity_with_uq(
    raw_capacity_curve,
    y_pred,
    lower,
    upper,
    vehicle_id,
    window_size=100,
    save_dir="plots_uq",
):
    os.makedirs(save_dir, exist_ok=True)

    total_curve = np.asarray(raw_capacity_curve, dtype=float)
    pred_x = np.arange(window_size, window_size + len(y_pred))

    plt.figure(figsize=(16, 10))
    plt.plot(np.arange(len(total_curve)), total_curve, "k-", linewidth=1.5, label="Real")
    plt.plot(pred_x, y_pred, "b-", linewidth=1.5, label="Model")
    plt.fill_between(pred_x, lower, upper, alpha=0.25, label="Prediction Interval")
    plt.axvline(x=window_size, c="r", ls="--", linewidth=1.2, label="Prediction Start")
    plt.title(f"Vehicle {vehicle_id}: Capacity Prediction with UQ")
    plt.ylabel("Capacity (Ah)")
    plt.xlabel("Cycle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vehicle_{vehicle_id}_uq_curve.png"), dpi=300)
    plt.show()

@torch.no_grad()
def predict_validation_vehicle_with_uq(
    model,
    val_loader,
    scaler,
    raw_capacity_curve,
    vehicle_id,
    device,
    alpha=0.1,
    uq_method="Quantile+Integrator(log)",
    uq_lr=0.01,
    T_burnin=30,
    uq_method_kwargs=None,
    plot_dir="plots_uq_val",
):
    if uq_method_kwargs is None:
        uq_method_kwargs = {
            "Csat": 3.0,
            "KI": 1.0,
            "proportional_lr": True,
        }

    capacity_scaler = float(scaler[-1])
    y_pred, y_true, stand_y_pred, stand_y_true = collect_predictions(model, val_loader, device, capacity_scaler)

    val_scores = absolute_residual_score(y_true, y_pred)

    uq_result = run_uq_method(
        scores=val_scores,
        method_name=uq_method,
        alpha=alpha,
        lr=uq_lr,
        ahead=1,
        T_burnin=T_burnin,
        method_kwargs=uq_method_kwargs,
    )

    q_val = np.maximum(np.asarray(uq_result["q"]), 0.0)
    lower = y_pred - q_val
    upper = y_pred + q_val

    # 计算各种测试指标
    mse = np.mean((stand_y_pred - stand_y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(stand_y_pred - stand_y_true))
    mape = mape_func(stand_y_true, stand_y_pred)
    r2 = r2_score(stand_y_true, stand_y_pred)

    uq_metrics = compute_uq_metrics(
        y_true=y_true,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        q=q_val,
        alpha=alpha,
    )

    plot_capacity_with_uq(
        raw_capacity_curve=np.asarray(raw_capacity_curve, dtype=float),
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        vehicle_id=f"val_{vehicle_id}",
        window_size=100,
        save_dir=plot_dir,
    )

    return {
        "point_metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
        },
        "uq_metrics": uq_metrics,
    }


# 主程序
def main():
    set_seed(42)

    # 1. 超参数
    window_size = 100
    forecast_horizon = 1
    batch_size = 64
    epochs = 15
    lr = 0.001
    weight_decay = 1e-5
    patience = 20

    # UQ 参数
    alpha = 0.05  # 95% CI
    # uq_method = "Quantile+Integrator(log)"
    # uq_method = "Quantile+Integrator(log)+Scorecaster"
    # uq_method="ECI"
    uq_method="ECI_cutoff"
    # uq_method='ECI_integral'
    # uq_method='full_smoothed_eci'
    # uq_method='OGD'
    # uq_method='SF_OGD'
    # uq_method='decay_OGD'
    uq_lr = 0.005
    T_burnin = 30
    uq_method_kwargs = {
        "Csat": 3.0,
        "KI": 1.0,
        "proportional_lr": True,
    }


    # 2. 数据路径
    dir_path = r"E:\My_EV_set_0"
    car_id = [i for i in range(1, 21)]

    # dir_path=r"E:\qinghua\battery_features"
    # car_id=[3, 15, 17, 24, 30, 34, 35, 37, 51, 58, 70, 88,
    #         92, 109, 132, 140, 153, 154, 166, 176, 52, 141, 57, 5, 177]

    all_veh_data = []
    veh_ca = []
    for cid in car_id:
        path = os.path.join(dir_path, f"#{cid}.csv")
        veh = pd.read_csv(path)
        data = copy.deepcopy(veh.values)
        ca = copy.deepcopy(veh.values[:, -1])
        all_veh_data.append(np.array(data))
        veh_ca.append(list(ca))


    # 3. 三划分
    train_idx = list(range(0, len(car_id)-2))
    val_idx = list(range(len(car_id)-2, len(car_id)-1))
    test_idx = list(range(len(car_id)-1, len(car_id)))

    train_data = [all_veh_data[i] for i in train_idx]
    val_data = [all_veh_data[i] for i in val_idx]

    scaler = compute_scaler(train_data)

    train_X, train_y, _ = prepare_multi_timestep_data(
        train_data, window_size=window_size, forecast_horizon=forecast_horizon, scaler=scaler)
    val_X, val_y, _ = prepare_multi_timestep_data(
        val_data, window_size=window_size, forecast_horizon=forecast_horizon, scaler=scaler)

    train_dataset = TimeSeriesDataset(train_X, train_y)
    val_dataset = TimeSeriesDataset(val_X, val_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 每辆测试车单独建 loader，保证时间顺序不打乱
    test_loaders = []
    for i in test_idx:
        test_X, test_y = prepare_single_vehicle_test_data(
            all_veh_data[i], scaler=scaler, window_size=window_size, forecast_horizon=forecast_horizon
        )
        test_dataset = TimeSeriesDataset(test_X, test_y)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        test_loaders.append((car_id[i], test_loader, veh_ca[i]))


    # 4. 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # model = Mamba_Agent(
    #     seq_len=window_size,
    #     in_channels=7,
    #     model_dim=64,
    #     d_state=64,
    #     drop=0.2,
    #     pool='mean',
    # ).to(device)

    # 用于联邦测试
    model = MambaAgentWithFedAWA(
        seq_len=100, in_channels=7, model_dim=64, d_state=64, drop=0.2, pool='mean').to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    '''
    # -------------------------
    # 5. 训练 + 验证集选最优模型
    # -------------------------
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_path="best_model_uq.pth",
        patience=patience,
    )

    torch.save(model.state_dict(), "final_model_uq.pth")
    torch.save(optimizer.state_dict(), "final_optimizer_uq.pth")
    print("训练完成")
    '''

    # 普通测试
    # state = torch.load(r"E:\PycharmProjects\my_battery_research\目前最佳模型参数\model15.pth", map_location=device)
    # model.load_state_dict(state, strict=True)

    # 联邦学习测试
    checkpoint = torch.load(r"/FedAWA_FNR/record_2/checkpoint\global_model_round_39.pth",
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])



    # 6. 验证集残差 -> 作为 warmup_scores
    warmup_scores = collect_calibration_scores(
        model=model,
        val_loader=val_loader,
        device=device,
        capacity_scaler=float(scaler[-1]),
    )


    # 7. 测试集：逐车输出 点预测 + 区间 + 指标 + 画图
    print('===== 画出测试集的不确定性量化 =====')
    all_t_results = []
    for veh_id, test_loader, raw_capacity_curve in test_loaders:
        result = predict_test_vehicle_with_uq(
            model=model,
            test_loader=test_loader,
            warmup_scores=warmup_scores,
            scaler=scaler,
            raw_capacity_curve=raw_capacity_curve,
            vehicle_id=veh_id,
            device=device,
            alpha=alpha,
            uq_method=uq_method,
            uq_lr=uq_lr,
            T_burnin=T_burnin,
            uq_method_kwargs=uq_method_kwargs,
            plot_dir="plots_uq",
        )
        all_t_results.append(result)

    print('===== 画出验证集的不确定性量化 =====')
    val_vehicle_loaders = []
    for i in val_idx:
        val_X_single, val_y_single = prepare_single_vehicle_test_data(
            all_veh_data[i],
            scaler=scaler,
            window_size=window_size,
            forecast_horizon=forecast_horizon
        )
        val_dataset_single = TimeSeriesDataset(val_X_single, val_y_single)
        val_loader_single = DataLoader(
            val_dataset_single,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        val_vehicle_loaders.append((car_id[i], val_loader_single, veh_ca[i]))

    val_results = []
    for veh_id, val_loader_single, raw_capacity_curve in val_vehicle_loaders:
        val_result = predict_validation_vehicle_with_uq(
            model=model,
            val_loader=val_loader_single,
            scaler=scaler,
            raw_capacity_curve=raw_capacity_curve,
            vehicle_id=veh_id,
            device=device,
            alpha=alpha,
            uq_method=uq_method,
            uq_lr=uq_lr,
            T_burnin=T_burnin,
            uq_method_kwargs=uq_method_kwargs,
            plot_dir="plots_uq_val",
        )
        val_results.append(val_result)


    # 8. 汇总测试集指标
    if len(all_t_results) > 0:
        point_keys = list(all_t_results[0]["point_metrics"].keys())
        uq_keys = list(all_t_results[0]["uq_metrics"].keys())

        print("\n===== Overall Test Summary =====")
        for k in point_keys:
            vals = [x["point_metrics"][k] for x in all_t_results]
            print(f"{k}: {np.mean(vals):.6f}")
        for k in uq_keys:
            vals = [x["uq_metrics"][k] for x in all_t_results]
            print(f"{k}: {np.mean(vals):.6f}")

    if len(val_results) > 0:
        point_keys = list(all_t_results[0]["point_metrics"].keys())
        uq_keys = list(val_results[0]["uq_metrics"].keys())

        print("\n===== Overall Validation Summary =====")
        for k in point_keys:
            vals = [x["point_metrics"][k] for x in val_results]
            print(f"{k}: {np.mean(vals):.6f}")
        for k in uq_keys:
            vals = [x["uq_metrics"][k] for x in val_results]
            print(f"{k}: {np.mean(vals):.6f}")


if __name__ == "__main__":
    main()

'''
联邦学习的 UQ 结果
ECI
Veh 20
===== Overall Test Summary =====
MAE: 0.003310
MSE: 0.000012
RMSE: 0.003514
R2: 0.994248
MAPE: 0.374377
Coverage(%): 89.247312
Average width: 1.260387
Median width: 1.220489
CRPS: 0.303660
PICP: 0.892473
MPICD: 0.448679

ECI-cutoff
Veh 177
===== Overall Test Summary =====
MAE: 0.002720
MSE: 0.000009
RMSE: 0.003013
R2: 0.990260
MAPE: 0.291672
Coverage(%): 89.017341
Average width: 0.376307
Median width: 0.378777
CRPS: 0.084301
PICP: 0.890173
MPICD: 0.122041

Veh 20
===== Overall Test Summary =====
MAE: 0.003310
MSE: 0.000012
RMSE: 0.003514
R2: 0.994248
MAPE: 0.374377
Coverage(%): 90.937020
Average width: 1.308322
Median width: 1.277080
CRPS: 0.301264
PICP: 0.909370
MPICD: 0.448679

'''




