import torch
from torch import nn as nn


class Clip(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.clamp(min=0, max=1)


class LInfClip(nn.Module):
    def __init__(self, original: torch.tensor, eps: float = 16 / 255):
        super().__init__()
        self.base = original.detach().clone().cuda()
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x + torch.clip(self.base - x, min=-self.eps, max=self.eps)


class L2Clip(nn.Module):
    def __init__(self, original: torch.tensor, eps: float = 16 / 255):
        super().__init__()
        self.base = original.detach().clone().cuda()
        self.eps = eps

    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        delta = self.base - x
        norm = delta.norm(p=2)
        delta = self.eps * delta / norm if norm > self.eps else delta
        return x + delta
