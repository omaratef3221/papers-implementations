import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, dff = 2048):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, dff, bias=True)
        self.W2 = nn.Linear(dff, d_model, bias = True)
    def forward(self, x):
        return self.W2(torch.relu(self.W1(x)))
    

