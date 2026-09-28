import torch
import copy
import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score

from fedawa import fedawa

def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    client_params = []  # 存储客户端模型参数
    for idx in select_list:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_params.append(client_nodes[idx].model.get_param(clone=True))
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    agg_weights = [size_weights[idx] for idx in select_list]
    return agg_weights, client_params  # 返回每个客户端的权重、模型参数

# 一开始权重是不知道的
global_T_weights = None   # 用于 fedawa 的全局权重

def Server_update(args, central_node, client_nodes, select_list, size_weights, rounds_num=None, change=0):
    global global_T_weights
    if rounds_num == change:
        global_T_weights = torch.tensor(size_weights, dtype=torch.float32).cuda()   # 改为张量

    # 接收客户端模型
    agg_weights, client_params = receive_client_models_pool(args, client_nodes, select_list, size_weights)

    if args.server_method == 'fedawa':
        T_weights = global_T_weights   # 已经是张量
        avg_global_param, cur_global_T_weight = fedawa(args, client_params, agg_weights,
                                                       central_node, rounds_num, T_weights)
        global_T_weights = cur_global_T_weight
        prob = torch.nn.functional.softmax(cur_global_T_weight, dim=0)
        for i, idx in enumerate(select_list):
            size_weights[idx] = cur_global_T_weight[i].item()
        central_node.model.load_param(avg_global_param)
    else:
        raise ValueError('Only fedawa is supported in this simplified version.')
    return central_node, prob  # 返回概率张量


