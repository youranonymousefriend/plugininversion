import torch
from torch import nn as nn

from datasets import Normalizer


class TotalVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_wise = x[:, :, :, 1:] - x[:, :, :, :-1]
        y_wise = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(2, 3)).mean() + y_wise.norm(p=self.p, dim=(2, 3)).mean() + \
               diag_1.norm(p=self.p, dim=(2, 3)).mean() + diag_2.norm(p=self.p, dim=(2, 3)).mean()


class L1Norm(nn.Module):
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.norm(p=1, dim=(1, 2, 3)).mean()


class L2Norm(nn.Module):
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.norm(p=2, dim=(1, 2, 3)).mean()


class FakeColorDistribution(nn.Module):
    def __init__(self, normalizer: Normalizer):
        super().__init__()
        self.normalizer = normalizer

    def forward(self, x: torch.tensor) -> torch.tensor:
        view = x.transpose(1, 0).contiguous().view([x.size(1), -1])
        mean, std = view.mean(-1), view.std(-1, unbiased=False)
        mean_loss = (mean.view(-1) - self.normalizer.mean.view(-1)).norm()
        std_loss = (std.view(-1) - self.normalizer.std.view(-1)).norm()
        return mean_loss + std_loss
