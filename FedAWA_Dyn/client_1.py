import torch
import torch.nn as nn
import copy
import numpy as np
from MAgent.Mamba_Agent import Mamba_Agent
from MAgent.Mamba_Agent import Mamba_Agent as BaseMambaAgent
from sklearn.metrics import r2_score
from torch.optim import Optimizer


# -------------------- Perturbed Gradient Descent (用于 FedProx) --------------------
class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, global_params):
        """
        global_params: 全局模型的参数列表（与本地模型顺序一致）
        """
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:
                    continue
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


class MambaAgentWithFedAWA(BaseMambaAgent):
    def get_param(self, clone=False):
        """返回包含所有参数和 flat_w 的字典"""
        state_dict = self.state_dict()
        param_dict = {}
        for key, tensor in state_dict.items():
            param_dict[key] = tensor.clone() if clone else tensor

        flat_w = torch.cat([p.data.view(-1) for p in self.parameters()])
        param_dict['flat_w'] = flat_w.clone() if clone else flat_w
        return param_dict

    def load_param(self, param_dict):
        """从 get_param 返回的字典加载参数（忽略 flat_w）"""
        state_dict = {k: v for k, v in param_dict.items() if k != 'flat_w'}
        self.load_state_dict(state_dict)


def init_model(model_type, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'Mamba_Agent':
        model = MambaAgentWithFedAWA(
            seq_len=args.window_size,
            in_channels=args.in_channels,
            model_dim=64,
            d_state=64,
            agent_num=20,
            drop=0.2,
            pool='mean'
        ).to(device)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def init_optimizer(num_id, model, args):
    # FedProx 使用自定义优化器
    if num_id > -1 and args.client_method == 'fedprox':
        return PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
    else:
        if args.optimizer == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.local_wd_rate
            )
        elif args.optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")


# -------------------- 节点类 --------------------
class Node:
    def __init__(self, num_id, local_data, validate_set, args, scaler=None):
        self.num_id = num_id
        self.args = args
        self.local_data = local_data
        self.validate_set = validate_set
        self.model = init_model(args.local_model, args)
        self.optimizer = init_optimizer(num_id, self.model, args)
        self.scaler = scaler

        # -------------------- FedDyn 状态 --------------------
        self.feddyn_y = None
        if num_id > -1:
            with torch.no_grad():
                flat_w = torch.cat([p.data.view(-1) for p in self.model.parameters()])
                self.feddyn_y = torch.zeros_like(flat_w)

    def zero_weights(self, model):
        for p in model.parameters():
            p.data.zero_()


# -------------------- FedDyn 辅助函数 --------------------
def get_model_flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


# -------------------- 客户端训练函数 --------------------

def client_localTrain(args, node, node_id):
    print('Starting client local training......')
    node.model.train()
    criterion_mse = nn.MSELoss()
    mae_loss = nn.L1Loss()
    last_mse, last_mae, last_rmse = 0.0, 0.0, 0.0
    train_loader = node.local_data
    device = next(node.model.parameters()).device

    for epoch in range(args.local_E[node_id]):
        epoch_mse, epoch_mae, epoch_rmse = 0.0, 0.0, 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            node.optimizer.zero_grad()

            output = node.model(data)
            loss_mse = criterion_mse(output, target)
            loss_mae = mae_loss(output, target)
            loss_rmse = torch.sqrt(loss_mse)

            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(node.model.parameters(), max_norm=1.0)
            node.optimizer.step()

            epoch_mse += loss_mse.item()
            epoch_mae += loss_mae.item()
            epoch_rmse += loss_rmse.item()

        avg_mse = epoch_mse / len(train_loader)
        avg_mae = epoch_mae / len(train_loader)
        avg_rmse = epoch_rmse / len(train_loader)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Client {node.num_id}  Epoch {epoch+1}/{args.local_E[node_id]}: "
                  f"MSE={avg_mse:.6f}, RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")

        if epoch == args.local_E[node_id] - 1:
            last_mse = avg_mse
            last_mae = avg_mae
            last_rmse = avg_rmse

    return last_mse, last_mae, last_rmse


def client_fedprox(global_model_param, args, node, node_id):
    print('Starting client local training with fedprox......')
    node.model.train()
    criterion_mse = nn.MSELoss()
    mae_loss = nn.L1Loss()
    last_mse, last_mae, last_rmse = 0.0, 0.0, 0.0
    train_loader = node.local_data
    device = next(node.model.parameters()).device

    for epoch in range(args.local_E[node_id]):
        epoch_mse, epoch_mae, epoch_rmse = 0.0, 0.0, 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            node.optimizer.zero_grad()

            output = node.model(data)
            loss_mse = criterion_mse(output, target)
            loss_mae = mae_loss(output, target)
            loss_rmse = torch.sqrt(loss_mse)

            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(node.model.parameters(), max_norm=1.0)
            node.optimizer.step(global_model_param)

            epoch_mse += loss_mse.item()
            epoch_mae += loss_mae.item()
            epoch_rmse += loss_rmse.item()

        avg_mse = epoch_mse / len(train_loader)
        avg_mae = epoch_mae / len(train_loader)
        avg_rmse = epoch_rmse / len(train_loader)

        print(f"Client {node.num_id}  Epoch {epoch+1}/{args.local_E[node_id]}: "
              f"MSE={avg_mse:.6f}, RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")

        if epoch == args.local_E[node_id] - 1:
            last_mse = avg_mse
            last_mae = avg_mae
            last_rmse = avg_rmse

    return last_mse, last_mae, last_rmse


def client_feddyn(global_model, args, node, node_id):
    """
    FedDyn 本地训练目标：
        loss = f_i(w) - <h_i, w> + (alpha / 2) * ||w - w_t||^2
    """
    print('Starting client local training with feddyn......')
    node.model.train()
    criterion_mse = nn.MSELoss()
    mae_loss = nn.L1Loss()
    last_mse, last_mae, last_rmse = 0.0, 0.0, 0.0
    train_loader = node.local_data
    device = next(node.model.parameters()).device

    alpha = args.feddyn_alpha[node_id] if isinstance(args.feddyn_alpha, (list, tuple)) else args.feddyn_alpha
    global_flat = torch.cat([p.detach().view(-1).to(device) for p in global_model.parameters()])

    if node.feddyn_y is None:
        node.feddyn_y = torch.zeros_like(global_flat)
    else:
        node.feddyn_y = node.feddyn_y.to(device)

    for epoch in range(args.local_E[node_id]):
        epoch_mse, epoch_mae, epoch_rmse = 0.0, 0.0, 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            node.optimizer.zero_grad()

            output = node.model(data)
            loss_mse = criterion_mse(output, target)
            loss_mae = mae_loss(output, target)
            loss_rmse = torch.sqrt(loss_mse)

            local_flat = get_model_flat_params(node.model)

            linear_term = torch.dot(node.feddyn_y, local_flat)
            quad_term = 0.5 * alpha * torch.sum((local_flat - global_flat) ** 2)

            total_loss = loss_mse - linear_term + quad_term
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(node.model.parameters(), max_norm=1.0)
            node.optimizer.step()

            epoch_mse += loss_mse.item()
            epoch_mae += loss_mae.item()
            epoch_rmse += loss_rmse.item()

        avg_mse = epoch_mse / len(train_loader)
        avg_mae = epoch_mae / len(train_loader)
        avg_rmse = epoch_rmse / len(train_loader)

        print(f"Client {node.num_id}  Epoch {epoch+1}/{args.local_E[node_id]}: "
              f"MSE={avg_mse:.6f}, RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")

        if epoch == args.local_E[node_id] - 1:
            last_mse = avg_mse
            last_mae = avg_mae
            last_rmse = avg_rmse

    with torch.no_grad():
        new_local_flat = get_model_flat_params(node.model).detach()
        node.feddyn_y = node.feddyn_y - alpha * (new_local_flat - global_flat)

    return last_mse, last_mae, last_rmse


def receive_server_model(args, client_nodes, central_node):
    for node in client_nodes:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            try:
                node.model.load_param(copy.deepcopy(central_node.model.get_param(clone=True)))
            except AttributeError:
                node.model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
        else:
            node.model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
    return client_nodes


def Client_update(args, client_nodes, central_node):
    client_nodes = receive_server_model(args, client_nodes, central_node)
    client_mse, client_mae, client_rmse = [], [], []

    for i in range(len(client_nodes)):
        node = client_nodes[i]

        if args.client_method == 'local_train':
            last_mse, last_mae, last_rmse = client_localTrain(args, node, i)

        elif args.client_method == 'fedprox':
            global_params = copy.deepcopy(list(central_node.model.parameters()))
            last_mse, last_mae, last_rmse = client_fedprox(global_params, args, node, i)

        elif args.client_method == 'feddyn':
            last_mse, last_mae, last_rmse = client_feddyn(central_node.model, args, node, i)

        else:
            raise ValueError('Undefined client method')

        client_mse.append(last_mse)
        client_mae.append(last_mae)
        client_rmse.append(last_rmse)

    return client_nodes, client_mse, client_mae, client_rmse


def Client_validate(args, client_nodes):
    client_mse, client_mae, client_rmse, client_r2, client_mape = [], [], [], [], []
    for node in client_nodes:
        result = validate(args, node, scaler=node.scaler, which_dataset='client')
        client_mse.append(result['mse'])
        client_mae.append(result['mae'])
        client_rmse.append(result['rmse'])
        client_r2.append(result['r2'])
        client_mape.append(result['mape'])
    return client_mse, client_mae, client_rmse, client_r2, client_mape


def validate(args, node, scaler, which_dataset='client'):
    print('Starting client validating')
    total_loader = []

    if which_dataset == 'client':
        loader1, loader2 = node.validate_set[0], node.validate_set[1]
        total_loader = [loader1, loader2]
    else:
        raise ValueError('Undefined...')

    result = {'mse': [], 'mae': [], 'rmse': [], 'r2': [], 'mape': []}
    device = next(node.model.parameters()).device

    with torch.no_grad():
        node.model.eval()
        for i in range(len(total_loader)):
            loader = total_loader[i]
            total_mse, total_mae, total_rmse = [], [], []
            stan_preds, stan_targets = [], []

            criterion_mse = nn.MSELoss()
            criterion_mae = nn.L1Loss()

            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = node.model(data)

                mse = criterion_mse(output, target)
                mae = criterion_mae(output, target)
                rmse = torch.sqrt(mse)

                total_mse.append(mse.item())
                total_mae.append(mae.item())
                total_rmse.append(rmse.item())

                capacity_scaler = scaler[-1]
                output_np = output.cpu().numpy()
                capacity_np = target.cpu().numpy()
                output_restored = output_np * capacity_scaler
                capacity_restored = capacity_np * capacity_scaler

                stan_preds.extend(output_restored.flatten().tolist())
                stan_targets.extend(capacity_restored.flatten().tolist())

            mean_mse = np.mean(total_mse)
            mean_mae = np.mean(total_mae)
            mean_rmse = np.mean(total_rmse)
            r2 = r2_score(stan_targets, stan_preds)
            mape = np.mean(
                np.abs((np.array(stan_targets) - np.array(stan_preds)) / (np.array(stan_targets) + 1e-8))
            ) * 100

            result['mse'].append(mean_mse)
            result['mae'].append(mean_mae)
            result['rmse'].append(mean_rmse)
            result['r2'].append(r2)
            result['mape'].append(mape)

    return result