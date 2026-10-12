import torch
import torch.nn as nn

import math

class EmbeddingBlock(nn.Module):
  def __init__(self, vocab_size, d_model):
    super(EmbeddingBlock, self).__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.sqrt_d_model = math.sqrt(d_model)
    self.embeddinglayer = nn.Embedding(vocab_size, d_model)

  def forward(self, x: torch.Tensor):
    return self.embeddinglayer(x) * self.sqrt_d_model
