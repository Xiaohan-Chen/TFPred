import torch
import torch.nn as nn

class CrossCorrelationLoss(nn.Module):
    def __init__(self, lambda_param = 5e-3):
        super(CrossCorrelationLoss, self).__init__()
        self.lambda_param = lambda_param
    def forward(self, x1, x2):
        device = x1.device

        x1_shape, x2_shape = x1.size(), x2.size()
        assert x1_shape == x2_shape # [N x D] [batch_size, dim]
        N, D = x1.shape
        # normalization
        x1 = (x1 - x1.mean(0)) / x1.std(0)
        x2 = (x2 - x2.mean(0)) / x2.std(0)

        # cross-correlation
        c = torch.mm(x1.T, x2) / N # [D x D]

        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss