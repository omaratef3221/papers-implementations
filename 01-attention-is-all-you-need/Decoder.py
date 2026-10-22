import torch.nn as nn
import torch

from Layers.MultiheadAttention import MultiHeadAttention
from Layers.LayerNorm import LayerNorm
from Layers.FeedForward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, D_MODEL, H, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_multiheadattention = MultiHeadAttention(D_MODEL, H)
        self.layernorm1 = LayerNorm(D_MODEL)
        self.cross_multiheadattention = MultiHeadAttention(D_MODEL, H)
        self.layernorm2 = LayerNorm(D_MODEL)
        self.feedforward = FeedForward(D_MODEL)
        self.layernorm3 = LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        masked_attn_output = self.masked_multiheadattention(x, 
                                                            x, 
                                                            x, 
                                                            mask = tgt_mask)
        
        add_norm1_output = self.layernorm1(masked_attn_output + x)

        cross_masked_attn_output = self.cross_multiheadattention(add_norm1_output, 
                                                                 encoder_output, 
                                                                 encoder_output, src_mask)
        add_norm2_output = self.layernorm2(self.dropout(cross_masked_attn_output) + add_norm1_output)

        ffn_output = self.feedforward(add_norm2_output)
        add_norm3_output = self.layernorm3(self.dropout(ffn_output) + add_norm2_output)

        return add_norm3_output