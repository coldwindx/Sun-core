import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, 1)
    def forward(self, x, mask=None):
        # x: [BatchSize, SeqLen, InputDim] -> [SeqLen, BatchSize, InputDim]
        x = x.permute(1, 0, 2)
        scores = self.U(torch.tanh(self.W(x)))
        weights = F.softmax(scores, dim=0)
        pooled = torch.sum(x * weights, dim=0)
        return pooled

class MaskedMeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        '''
            x:      [BatchSize, SeqLen, Dim]
            mask:   [BatchSize, SeqLen]
        '''
        return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(1)

if __name__ == "__main__":
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 0]
    ])

    x = torch.ones((2, 4, 3))

    pool = MaskedMeanPooling()
    x = pool.forward(x, mask)
    print(x)