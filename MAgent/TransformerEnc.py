# transformer_encoder_agent.py
import torch
from torch import nn
from MAgent.new_Agent import AgentAttention
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class AgentTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, seq_len, num_heads=4, sr_ratio=2, agent_num=25,
                 mlp_ratio=4.0, drop=0.2, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = AgentAttention(
            dim=dim, seq_len=seq_len, num_heads=num_heads,
            sr_ratio=sr_ratio, agent_num=agent_num,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.drop_path = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim))

    def forward(self, x):
        # x: [B, N, dim]
        # model 14/15
        res=x
        x = self.attn(x)
        x = self.norm(res+self.mlp(x))


        '''
        # model 4/5/6:
        res=x
        x=self.attn(x)
        x=self.norm(res+self.mlp(x))
        z=self.branch_mamba(agent_out+mamba_out)
        
        # model 7/8/9:
        res=x
        x=self.attn(x)
        x=self.norm(x+self.mlp(x))
        z=self.branch_mamba(agent_out+mamba_out)
        
        # model 10/11/12:
        x=self.attn(x)
        x=self.norm(x+self.attn(x))
        x=self.norm(x+self.mlp(x))
        '''
        
        # model 13:
        # x = self.norm(x+self.attn(x))
        # x = self.mlp(x)
        return x

class AgentTransformerEncoder(nn.Module):
    def __init__(self, dim, seq_len, depth=2, num_heads=4, sr_ratio=2, agent_num=25,
                 mlp_ratio=4.0, drop=0.2, attn_drop=0.2, proj_drop=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.pe = PositionalEncoding(dim)
        layers = []
        for _ in range(depth):
            layers.append(
                AgentTransformerEncoderLayer(
                    dim=dim, seq_len=seq_len, num_heads=num_heads,
                    sr_ratio=sr_ratio, agent_num=agent_num,
                    mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, proj_drop=proj_drop
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, N, dim]
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return x  # [B, N, dim]

