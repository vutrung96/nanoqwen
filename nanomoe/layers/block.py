import torch.nn as nn

from nanomoe.config import MoeConfig
from nanomoe.layers.attention import Attention, Rope
from nanomoe.layers.moe import Moe


class Block(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()

        self.config = config
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.rope = Rope(config)
        self.attention = Attention(config)
        self.moe = Moe(config)

    def forward(self, x, mask):
        sin, cos = self.rope(x)
        x = x + self.attention(self.norm1(x), sin, cos, mask)
        x = x + self.moe(self.norm2(x))
        return x
