import torch.nn as nn
from torch import Tensor
import torch

class totalloss(nn.Module):
    def __init__(self):
        super(totalloss, self).__init__()

    def forward(self, pred: Tensor, truth: Tensor, cluster_loss: Tensor, mask: Tensor):
        mask = mask
        mask = mask.gt(0).view(-1) # view(-1): to one dimension

        pred = torch.masked_select(pred.view(-1), mask)
        truth = torch.masked_select(truth.view(-1), mask)

        loss = torch.nn.functional.binary_cross_entropy(pred, truth.float(), reduction="mean") + 0.001 * cluster_loss
        return loss