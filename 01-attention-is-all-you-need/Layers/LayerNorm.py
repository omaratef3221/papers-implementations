import torch
import torch.nn as nn

## Non Learnable layernorm
class LayerNorm(nn.Module):
    def __init__(self, D_MODEL, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(D_MODEL))
        self.bias = nn.Parameter(torch.zeros(D_MODEL))
        self.eps = eps

    def forward(self, x):
        sigma = torch.std(x, dim = -1, unbiased=False).unsqueeze(-1)
        mean = torch.mean(x, dim = -1).unsqueeze(-1)
        
        numerator = x - mean
        result = torch.divide(numerator, torch.sqrt(sigma**2 + self.eps))
        return result * self.alpha + self.bias