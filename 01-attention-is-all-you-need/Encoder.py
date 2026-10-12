import torch.nn as nn
import torch

from Layers.MultiheadAttention import MultiHeadAttention
from Layers.LayerNorm import LayerNorm
from Layers.FeedForward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, D_MODEL, H):
        super(EncoderBlock, self).__init__()
        self.multiheadattention = MultiHeadAttention(D_MODEL, H)
        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()
        self.feedforward = FeedForward(D_MODEL)

    def forward(self, x, src_mask):
        attention_outputs = self.multiheadattention(x, x, x, mask = src_mask)
        layer_norm1 = self.layernorm1(attention_outputs + x)

        ffn_output = self.feedforward(layer_norm1)
        final_x = self.layernorm2(ffn_output + layer_norm1)
        return final_x