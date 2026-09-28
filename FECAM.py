# fecb.py
from distutils.command.config import config
import torch.nn as nn
import math
import numpy as np
import torch

def rfft(x, d):
    t = torch.fft.fft(x, dim=(-d))
    r = torch.stack((t.real, t.imag), -1)
    return r

def irfft(x, d):
    t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
    return t.real

def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

class dct_channel_block(nn.Module):
    def __init__(self, channel_len):  # channel_len 指通道长度，也就是时间序列长度
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_len, channel_len * 2, bias=False),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(channel_len * 2, channel_len, bias=False),
            nn.Sigmoid()
        )
        # 原始代码中的 LayerNorm([96]) 替换为 LayerNorm([channel_len]) 以适配实际长度
        self.dct_norm = nn.LayerNorm([channel_len], eps=1e-6)

    def forward(self, x):
        """
        x: [B, D, L]
        返回: lr_weight [B, D, L]
        """
        b, c, L = x.size()
        lst = []
        for i in range(c):
            freq = dct(x[:, i, :])
            lst.append(freq)

        stack_dct = torch.stack(lst, dim=1)  # [B,c,L]

        # 原始结构：两次LayerNorm，中间穿插fc
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(lr_weight)
        lr_weight = self.dct_norm(lr_weight)

        return x*lr_weight


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.rand(64, 7, 100).to(device)
    model=dct_channel_block(100).to(device)
    result = model(tensor)
    print("result.shape:", result.shape)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"Total params: {total}, Trainable: {trainable}, Memory: {mem_mb:.2f} MB")
