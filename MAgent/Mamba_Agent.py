import torch
from torch import nn
from FECAM import dct_channel_block
from mamba_ssm import Mamba
from MAgent.TransformerEnc import AgentTransformerEncoderLayer
from MAgent.new_Agent import AgentAttention

class FECAMBlock(nn.Module):
    def __init__(self, seq_len, dim, dropout=0.2):
        super(FECAMBlock,self).__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.fecam = dct_channel_block(channel_len=seq_len)
        # self.proj = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x):
        # x: [B, L, D]
        # FECAM 期望 [B, D, L]
        x_ch = x.permute(0, 2, 1).contiguous()  # [B, D, L]
        y_ch = self.fecam(x_ch)  # [B, D, L]
        y = y_ch.permute(0, 2, 1).contiguous()  # [B, L, D]
        # y = self.proj(y)
        # out = self.norm(x + y)
        return y

class Mamba_Agent(nn.Module):
    def __init__(self,seq_len=100, in_channels=7, model_dim=64, d_state=64,drop=0.2, agent_num=25, pool='mean'):
        super(Mamba_Agent,self).__init__()

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.pool = pool

        # 输入线性映射到 D
        self.in_proj = nn.Linear(in_channels, model_dim)
        self.branch_agent=AgentTransformerEncoderLayer(dim=model_dim, seq_len=seq_len, num_heads=2,
                                                       sr_ratio=2, agent_num=agent_num,
                                                       mlp_ratio=2.0, drop=0.2,
                                                       attn_drop=0.2, proj_drop=0.2)  # 用于联邦
        # self.gate = nn.Sequential(
        #     nn.Linear(model_dim, model_dim*2),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(model_dim*2, model_dim),
        #     nn.Sigmoid()
        # )
        self.branch_mamba1 = Mamba(d_model=model_dim, d_state=d_state)
        self.branch_mamba2= Mamba(d_model=model_dim, d_state=d_state)

        # FECAM堆叠（在通道拼接后的维度 2D 上）
        self.fcm_block = FECAMBlock(seq_len=seq_len, dim=model_dim, dropout=drop)

        # 最终回归头
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim//2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(model_dim//2, 1))

    def temporal_pool(self, x):
        # x: [B, L, D] -> [B, D]
        if self.pool == 'mean':
            return x.mean(dim=1)
        elif self.pool == 'last':
            return x[:, -1, :]
        elif self.pool == 'max':
            return x.max(dim=1).values
        else:
            return x.mean(dim=1)

    def forward(self, x):
        # x: [B, L, C]
        x = self.in_proj(x)                    # [B, L, D]
        mamba_out = self.branch_mamba1(x)       # [B, L, D]
        # agent_out=self.agent_encoder(mamba_out)
        agent_out = self.branch_agent(mamba_out)  # 用于联邦
        z = self.branch_mamba2(agent_out)
        z=self.fcm_block(z)   # 有 FECAM

        # 无 FECAM
        # gate_weight = self.gate(z)
        # z = z * gate_weight

        z_pool = self.temporal_pool(z)         # [B, D]
        y = self.head(z_pool)                  # [B, 1]
        return y

if __name__ == "__main__":
    # 简单自测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Mamba_Agent(
        seq_len=100, in_channels=7, model_dim=64,d_state=64,drop=0.2, pool='mean'
    ).to(device)

    x = torch.randn(64, 100, 7).to(device)
    y = model(x)
    print("Output shape:", y.shape)  # [64, 1]

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"Total params: {total}, Trainable: {trainable}, Memory: {mem_mb:.2f} MB")
    print("Output shape:", y.shape)  # 期望 [B, 1]

'''
Output shape: torch.Size([64, 1])
Total params: 199881, Trainable: 199881, Memory: 0.76 MB
'''

'''
Output shape: torch.Size([64, 1])
Total params: 216457, Trainable: 216457, Memory: 0.83 MB
Output shape: torch.Size([64, 1])
'''