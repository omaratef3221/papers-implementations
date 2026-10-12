import torch
import torch.nn as nn

## Non Learnable layernorm
class LayerNorm(nn.Module):
    def __init__(self, eps = 10e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        sigma = torch.std(x, dim = -1).unsqueeze(-1)
        mean = torch.mean(x, dim = -1).unsqueeze(-1)
        
        numerator = x - mean
        return torch.divide(numerator, torch.sqrt(sigma**2 + self.eps))