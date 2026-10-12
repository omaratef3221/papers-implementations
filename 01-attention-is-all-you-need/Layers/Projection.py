import torch
import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, D_MODEL, VOCAB_SIZE):
        super(Projection, self).__init__()
        self.linear = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        return self.linear(x)
