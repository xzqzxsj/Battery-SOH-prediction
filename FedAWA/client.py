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
                # 近端项修正：梯度 + mu * (p - g)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])

class MambaAgentWithFedAWA(BaseMambaAgent):
    def get_param(self, clone=False):
        """返回包含所有参数和 flat_w 的字典"""
        state_dict = self.state_dict()
        param_dict = {}
        for key, tensor in state_dict.items():
            param_dict[key] = tensor.clone() if clone else tensor
        # 添加展平向量
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
            drop=0.2,
            pool='mean'
        ).to(device)   # 移动到 cuda 运行模型
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def init_optimizer(num_id, model, args):  # 初始化客户端的优化器
    if num_id > -1 and args.client_method == 'fedprox':
        # 使用自定义的 PerturbedGradientDescent
        return PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
    else:
        if args.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
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
        self.local_data = local_data  # local_data 就是本地训练集
        self.validate_set = validate_set  # 本地验证集
        self.model = init_model(args.local_model, args)
        self.optimizer = init_optimizer(num_id, self.model, args)  # 初始化优化器
        self.scaler = scaler

    def zero_weights(self, model):
        for p in model.parameters():
            p.data.zero_()

# -------------------- 客户端训练函数 --------------------
def client_localTrain(args, node,node_id):
    print('Starting client local training......')
    node.model.train()
    criterion_mse = nn.MSELoss()
    mae_loss=nn.L1Loss()
    last_mse, last_mae, last_rmse = 0.0, 0.0, 0.0
    train_loader = node.local_data

    for epoch in range(args.local_E[node_id]):  # 每个客户端是不一样的 epoch 用于训练
        epoch_mse,epoch_mae,epoch_rmse = 0.0,0.0,0.0
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()

            node.optimizer.zero_grad()

            output = node.model(data)
            loss_mse = criterion_mse(output, target)
            loss_mae=mae_loss(output,target)
            loss_rmse = torch.sqrt(loss_mse)

            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(node.model.parameters(), max_norm=1.0)
            node.optimizer.step()

            epoch_mse += loss_mse.item()
            epoch_mae += loss_mae.item()
            epoch_rmse += loss_rmse.item()

        avg_mse = epoch_mse / len(train_loader)
        avg_mae = epoch_mae/ len(train_loader)
        avg_rmse = epoch_rmse / len(train_loader)
        if epoch==0 or (epoch+1)%10==0:
            print(f"Client {node.num_id}  Epoch {epoch+1}/{args.local_E[node_id]}: MSE={avg_mse:.6f}, RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")

        if epoch == args.local_E[node_id]-1: # 已经是训练的最后一轮了
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

    for epoch in range(args.local_E[node_id]):   # 每个客户端是不一样的 epoch 用于训练
        epoch_mse,epoch_mae,epoch_rmse = 0.0,0.0,0.0
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()

            node.optimizer.zero_grad()

            output = node.model(data)
            loss_mse = criterion_mse(output, target)
            loss_mae = mae_loss(output, target)
            loss_rmse = torch.sqrt(loss_mse)

            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(node.model.parameters(), max_norm=1.0)
            node.optimizer.step(global_model_param)  # 传入全局参数, 与 local_train 不一样的地方

            epoch_mse += loss_mse.item()
            epoch_mae += loss_mae.item()
            epoch_rmse += loss_rmse.item()

        avg_mse = epoch_mse / len(train_loader)
        avg_mae = epoch_mae / len(train_loader)
        avg_rmse = epoch_rmse / len(train_loader)

        print(f"Client {node.num_id}  Epoch {epoch+1}/{args.local_E[node_id]}: MSE={avg_mse:.6f}, RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")

        if epoch == args.local_E[node_id] - 1:  # 已经是训练的最后一轮了
            last_mse = avg_mse
            last_mae = avg_mae
            last_rmse = avg_rmse
    return last_mse, last_mae, last_rmse

def receive_server_model(args, client_nodes, central_node):
    for node in client_nodes:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            # 若模型支持 get_param/load_param 则使用，否则回退到 state_dict
            try:
                node.model.load_param(copy.deepcopy(central_node.model.get_param(clone=True)))
            except AttributeError:
                node.model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
        else:
            node.model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
    return client_nodes

def Client_update(args, client_nodes, central_node):  # 客户端更新训练
    client_nodes = receive_server_model(args, client_nodes, central_node)
    client_mse, client_mae, client_rmse = [],[],[]
    for i in range(len(client_nodes)):  # i 为客户端的 id
        node=client_nodes[i]
        if args.client_method == 'local_train':
            last_mse, last_mae, last_rmse = client_localTrain(args, node, i)
        elif args.client_method == 'fedprox':
            global_params = copy.deepcopy(list(central_node.model.parameters()))
            last_mse, last_mae, last_rmse = client_fedprox(global_params, args, node, i)
        else:
            raise ValueError('Undefined client method')
        client_mse.append(last_mse)
        client_mae.append(last_mae)
        client_rmse.append(last_rmse)

    return client_nodes, client_mse,client_mae, client_rmse

def Client_validate(args, client_nodes):   # 客户端训练后验证
    client_mse, client_mae, client_rmse, client_r2, client_mape = [], [], [], [], []
    for node in client_nodes:
        # 注意这里选择 validate
        result = validate(args, node, scaler=node.scaler,which_dataset='client')
        client_mse.append(result['mse'])
        client_mae.append(result['mae'])
        client_rmse.append(result['rmse'])
        client_r2.append(result['r2'])
        client_mape.append(result['mape'])
    return client_mse, client_mae, client_rmse, client_r2, client_mape

# -------------------- 验证函数 （用于客户端训练后验证和服务端测试） --------------------
def validate(args, node, scaler, which_dataset='client'):
    print('Starting client validating')
    total_loader=[]
    if which_dataset == 'client':  # 用于服务端验证
        loader1,loader2 = node.validate_set[0], node.validate_set[1]  # 服务端分数据集验证
        total_loader=[loader1,loader2]
    else:
        raise ValueError('Undefined...')

    result={'mse':[],'mae':[],'rmse':[],'r2':[],'mape':[]}
    with torch.no_grad():
        node.model.eval()
        for i in range(len(total_loader)):
            loader=total_loader[i]
            total_mse, total_mae, total_rmse = [],[],[]
            stan_preds, stan_targets=[],[]
            criterion_mse = nn.MSELoss()
            criterion_mae = nn.L1Loss()
            for data, target in loader:  # 测试集
                data, target = data.cuda(), target.cuda()
                output = node.model(data)
                mse = criterion_mse(output, target)
                mae = criterion_mae(output, target)
                rmse = torch.sqrt(mse)
                total_mse.append(mse.item())
                total_mae.append(mae.item())
                total_rmse.append(rmse.item())


                capacity_scaler=scaler[-1]
                output_np = output.cpu().numpy()
                capacity_np = target.cpu().numpy()
                output_restored = output_np * capacity_scaler
                capacity_restored = capacity_np * capacity_scaler

                stan_preds.extend(output_restored.flatten().tolist())
                stan_targets.extend(capacity_restored.flatten().tolist())

            mean_mse = np.mean(total_mse)
            result['mse'].append(mean_mse)
            mean_mae = np.mean(total_mae)
            result['mae'].append(mean_mae)
            mean_rmse = np.mean(total_rmse)
            result['rmse'].append(mean_rmse)
            r2 = r2_score(stan_targets,stan_preds)
            result['r2'].append(r2)
            mape = np.mean(np.abs((np.array(stan_targets) - np.array(stan_preds)) / (np.array(stan_targets) + 1e-8))) * 100
            result['mape'].append(mape)

    # 返回 5 个指标
    return result