import torch
import torch.nn as nn
from torch.nn import functional as F


class MAE(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        loss = F.l1_loss(preds, targets)
        return loss
