import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, max_len, d_model, dropout=0.1):
    super(PositionalEncoding, self).__init__()
    self.max_len = max_len
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)

    pos_matrix = torch.zeros(max_len, self.d_model)
    i = torch.arange(0, d_model, step = 2).unsqueeze(0)
    pos = torch.arange(0, max_len).unsqueeze(1)

    division_term = 10000**((i)/d_model)

    pos_matrix[:, ::2] = torch.sin(pos/division_term)
    pos_matrix[:, 1::2] = torch.cos(pos/division_term)

    self.register_buffer("pos_matrix", pos_matrix.unsqueeze(0))


  def forward(self, x):
    return self.dropout(x + self.pos_matrix[:, :x.shape[1],:])