import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy

def to_var(x, requires_grad=True):
    if isinstance(x, dict):
        return {k: to_var(v, requires_grad) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)
    else:
        return x

def _cost_matrix(x, y, dis, p=2):
    d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    if dis == 'cos':
        C = 1 - d_cosine(x_col, y_lin)
    elif dis == 'euc':
        C = torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
    else:
        raise ValueError('Unknown distance')
    return C

def fedawa(args, parameters, list_nums_local_data, central_node, rounds, global_T_weight):
    """
    FedAWA 聚合算法
    parameters: 客户端的模型参数列表（每个元素是 state_dict 或包含 'flat_w' 的字典）
    list_nums_local_data: 客户端的样本数量（此处可能用不到，但保留）
    central_node: 中央节点
    rounds: 当前轮数
    global_T_weight: 上一轮的聚合权重 tensor
    """
    # 确保 global_T_weight 是张量
    if not isinstance(global_T_weight, torch.Tensor):
        global_T_weight = torch.tensor(global_T_weight, dtype=torch.float32).cuda()

    # 获取中心模型的参数（假设模型有 get_param 方法）
    try:
        param = central_node.model.get_param()
        global_params = copy.deepcopy(param)
    except AttributeError:
        # 若没有 get_param，则用 state_dict 并展平
        # 简化：假设 parameters 中每个元素包含 'flat_w'，且 central_node 也支持 flat_w
        raise NotImplementedError("This simplified fedawa expects models with 'flat_w' attribute.")

    # 提取客户端 flat_w
    flat_w_list = [p['flat_w'] for p in parameters]
    local_param_list = torch.stack(flat_w_list)

    T_weights = to_var(global_T_weight)

    # 服务器优化器
    if args.server_optimizer == 'sgd':
        opt = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
    else:  # adam
        opt = optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))

    print("T_weights before update:", torch.nn.functional.softmax(T_weights, dim=0))

    for i in range(args.server_epochs):
        print("Starting server weight update:", i)
        prob = torch.nn.functional.softmax(T_weights, dim=0)  # 初始化权重

        # 正则损失：与中心模型的差异
        C = _cost_matrix(global_params['flat_w'].detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)
        reg_loss = torch.sum(prob * C, dim=(-2, -1))
        print("reg_loss:", reg_loss)

        # 相似度损失：客户端梯度与加权平均梯度的距离
        client_grad = local_param_list - global_params['flat_w']
        column_sum = torch.matmul(prob.unsqueeze(0), client_grad)  # weighted sum
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)
        sim_loss = torch.sum(prob * l2_distance, dim=(-2, -1))
        print("sim_loss:", sim_loss)

        loss = sim_loss + reg_loss

        # 学习最小化的损失
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("step", i, "Loss:", loss.item())

    global_T_weight = T_weights.data
    print("T_weights after update:", global_T_weight)
    prob = torch.nn.functional.softmax(T_weights, dim=0)
    print("probability after update:", prob)

    # 加权聚合得到新的全局参数
    # 假设 parameters 是 state_dict 列表，且每个项包含 'flat_w' 和其他参数
    # 我们按 prob 加权平均所有参数
    fedavg_global_params = copy.deepcopy(parameters[0])
    for key in parameters[0].keys():
        if key == 'flat_w':
            # flat_w 已在前面计算，可以跳过或重新计算
            continue
        list_values = []
        for p, w in zip(parameters, prob):
            list_values.append(p[key] * w * args.gamma)
        fedavg_global_params[key] = sum(list_values) / sum(prob)

    return fedavg_global_params, global_T_weight