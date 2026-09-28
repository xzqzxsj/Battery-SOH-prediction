import torch
from torch import nn
from timm.models.layers import to_2tuple, trunc_normal_
from einops.einops import rearrange

# 1D Attention
class AgentAttention(nn.Module):
    def __init__(self, dim, seq_len, num_heads=1, qkv_bias=True, attn_drop=0.2, proj_drop=0.2,
             sr_ratio=1, agent_num=25, bias_base_len_q=128, bias_base_len_k=128):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)  # 下采样
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.pool = nn.AdaptiveAvgPool1d(output_size=agent_num)

        # 1d 深度可分离卷积
        self.dwc = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)

        # 注意：将时间长度作为可插值的最后一维
        # agent -> K/V 的偏置: [h, A, base_k] -> 插值到 N'
        self.an_bias_1d = nn.Parameter(torch.zeros(num_heads, agent_num, bias_base_len_k))
        # Q -> agent 的偏置: [h, A, base_q] -> 插值到 N 后再换轴到 [B, h, N, A]
        self.na_bias_1d = nn.Parameter(torch.zeros(num_heads, agent_num, bias_base_len_q))
        trunc_normal_(self.an_bias_1d, std=.02)
        trunc_normal_(self.na_bias_1d, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape
        h = self.num_heads
        dh = c // h

        q = self.q(x)  # [B, N, C]

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1)  # [B, C, N]
            x_ = self.sr(x_)  # [B, C, N']
            n_prime = x_.shape[-1]
            x_ = x_.permute(0, 2, 1)  # [B, N', C]
            x_ = self.norm(x_)
            kv = self.kv(x_)  # [B, N', 2C]
        else:
            kv = self.kv(x)  # [B, N, 2C]
            n_prime = n

        kv = kv.view(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]  # [B, N', C]

        agent_tokens = self.pool(q.permute(0, 2, 1)).permute(0, 2, 1)  # [B, A, C]

        q = q.view(b, n, h, dh).permute(0, 2, 1, 3)  # [B, h, N, dh]
        k = k.view(b, n_prime, h, dh).permute(0, 2, 1, 3)  # [B, h, N', dh]
        v = v.view(b, n_prime, h, dh).permute(0, 2, 1, 3)  # [B, h, N', dh]
        agent_tokens = agent_tokens.view(b, self.agent_num, h, dh).permute(0, 2, 1, 3)  # [B, h, A, dh]

        # 插值到目标长度（F.interpolate的输入为 [N, C, L]）
        an_bias = nn.functional.interpolate(  # [h, A, base_k] -> [h, A, N']
            self.an_bias_1d, size=n_prime, mode='linear', align_corners=False
        ).unsqueeze(0).repeat(b, 1, 1, 1)  # [B, h, A, N']

        na_bias = nn.functional.interpolate(  # [h, A, base_q] -> [h, A, N]
            self.na_bias_1d, size=n, mode='linear', align_corners=False
        ).unsqueeze(0).repeat(b, 1, 1, 1)  # [B, h, A, N]
        na_bias = na_bias.permute(0, 1, 3, 2)  # [B, h, N, A]

        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + an_bias)  # [B, h, A, N']
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v  # [B, h, A, dh]

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + na_bias)  # [B, h, N, A]
        q_attn = self.attn_drop(q_attn)
        x_out = q_attn @ agent_v  # [B, h, N, dh]
        x_out = x_out.transpose(1, 2).reshape(b, n, c)  # [B, N, C]

        v_res = v.transpose(1, 2).reshape(b, n_prime, c).permute(0, 2, 1)  # [B, C, N']
        if self.sr_ratio > 1 and n_prime != n:
            v_res = nn.functional.interpolate(v_res, size=n, mode='linear', align_corners=False)  # [B, C, N]
        v_res = self.dwc(v_res)  # [B, C, N]
        x_out = x_out + v_res.permute(0, 2, 1)  # [B, N, C]

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    x = torch.randn(64, 100, 7).to(device)
    model = AgentAttention(dim=7, seq_len=100, num_heads=1,
                        sr_ratio=2,  # （时间下采样一半，N’=50)
                        agent_num=25, #（A=25）
                        bias_base_len_q=128, bias_base_len_k=128).to(device)
    out = model(x)

    print(out.shape)