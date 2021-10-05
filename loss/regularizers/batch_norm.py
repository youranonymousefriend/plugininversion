import torch
from torch import nn as nn

from datasets.base import EasyDataset
import os


class FakeBatchNorm(nn.Module):
    def __init__(self, model: nn.Module, dataset: EasyDataset, checkpoint: str = None):
        super().__init__()
        self.normalizer = dataset.normalizer

        if checkpoint is not None:
            path = os.path.join('checkpoints', 'reg', checkpoint)
            model.load_state_dict(torch.load(path))
            self.conv = model.conv
            self.bn = model.bn
        else:
            self.conv = model.conv
            self.bn = nn.BatchNorm2d(self.conv.out_channels)
        self.eval()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(self.normalizer(x))
        view = x.transpose(1, 0).contiguous().view([x.size(1), -1])
        mean, var = view.mean(1), view.var(1, unbiased=False)
        ret_val = torch.norm(self.bn.running_var.data - var, 2) + torch.norm(self.bn.running_mean.data - mean, 2)
        return ret_val
