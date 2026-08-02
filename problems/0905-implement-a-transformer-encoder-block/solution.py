import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: norm1, attn, norm2, mlp, dropout
        # pass
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # TODO: pre-LN attention sublayer, then pre-LN MLP sublayer, both with residual
        # pass
        x = x + self.dropout(self.attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x)
            )[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
