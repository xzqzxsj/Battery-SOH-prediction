'''
import torch
import copy
import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score
from fedawa import fedawa

def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_params.append(client_nodes[idx].model.get_param(clone=True))
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    agg_weights = [size_weights[idx] for idx in select_list]
    return agg_weights, client_params


# 一开始权重是不知道的
global_T_weights = None

def Server_update(args, central_node, client_nodes, select_list, size_weights, rounds_num=None, change=0):
    global global_T_weights

    if rounds_num == change:
        device = next(central_node.model.parameters()).device
        global_T_weights = torch.tensor(size_weights, dtype=torch.float32, device=device)

    # 接收客户端模型
    agg_weights, client_params = receive_client_models_pool(args, client_nodes, select_list, size_weights)

    if args.server_method == 'fedawa':
        T_weights = global_T_weights
        avg_global_param, cur_global_T_weight = fedawa(
            args,
            client_params,
            agg_weights,
            central_node,
            rounds_num,
            T_weights
        )
        global_T_weights = cur_global_T_weight
        prob = torch.nn.functional.softmax(cur_global_T_weight, dim=0)

        for i, idx in enumerate(select_list):
            size_weights[idx] = cur_global_T_weight[i].item()

        central_node.model.load_param(avg_global_param)
    else:
        raise ValueError('Only fedawa is supported in this simplified version.')
    return central_node, prob
'''

import torch
import copy
import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score
from fedawa import fedawa


def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    client_params = []

    for idx in select_list:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_params.append(client_nodes[idx].model.get_param(clone=True))
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))

    agg_weights = [size_weights[idx] for idx in select_list]
    return agg_weights, client_params


def fedavg_aggregate(client_params, agg_weights):
    """
    FedAvg aggregation.

    client_params:
        list of state_dict from selected clients

    agg_weights:
        list of aggregation weights, usually proportional to local dataset size
    """
    weights = torch.tensor(agg_weights, dtype=torch.float32)

    # normalize weights
    weights = weights / weights.sum()

    avg_params = copy.deepcopy(client_params[0])

    for key in avg_params.keys():
        avg_params[key] = 0.0

        for client_state, w in zip(client_params, weights):
            w = w.to(client_state[key].device)
            avg_params[key] += client_state[key] * w

    return avg_params, weights


# FedAWA needs this global variable
global_T_weights = None


def Server_update(args, central_node, client_nodes, select_list, size_weights, rounds_num=None, change=0):
    global global_T_weights

    # receive selected client models
    agg_weights, client_params = receive_client_models_pool(
        args, client_nodes, select_list, size_weights
    )

    # ---------------- FedAWA ----------------
    if args.server_method == 'fedawa':
        if rounds_num == change or global_T_weights is None:
            device = next(central_node.model.parameters()).device
            global_T_weights = torch.tensor(agg_weights, dtype=torch.float32, device=device)

        T_weights = global_T_weights

        avg_global_param, cur_global_T_weight = fedawa(
            args,
            client_params,
            agg_weights,
            central_node,
            rounds_num,
            T_weights
        )

        global_T_weights = cur_global_T_weight
        prob = torch.nn.functional.softmax(cur_global_T_weight, dim=0)

        for i, idx in enumerate(select_list):
            size_weights[idx] = cur_global_T_weight[i].item()

        central_node.model.load_param(avg_global_param)

    # ---------------- FedAvg ----------------
    elif args.server_method == 'fedavg':
        avg_global_param, prob = fedavg_aggregate(client_params, agg_weights)

        central_node.model.load_state_dict(avg_global_param)

        device = next(central_node.model.parameters()).device
        prob = prob.to(device)

    else:
        raise ValueError(f'Unsupported server method: {args.server_method}')

    return central_node, prob