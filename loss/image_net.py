import torch

from datasets import Normalizer
# from .hooks import MatchModelBNStatsHook
from .base import AbstractLoss
from .regularizers import TotalVariation as BaseTotalVariation, FakeColorDistribution as AbstractColorDistribution, \
    FakeBatchNorm as BaseFakeBN
from hooks import TimedHookHolder
import numpy as np


class FakeBatchNorm(AbstractLoss):
    def __init__(self, bn: BaseFakeBN, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.bn = bn

    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.bn(x)


class TotalVariation(AbstractLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseTotalVariation(p)
        self.size = size * size


class CrossEntropyLoss(AbstractLoss):
    def loss(self, x: torch.tensor):
        return self.xent(self.model(x), self.label)

    def __init__(self, model: torch.nn.Module, label: torch.tensor, coefficient: float = 1.):
        super().__init__(coefficient)
        self.model = model
        self.label = label
        self.xent = torch.nn.CrossEntropyLoss()


### Deprecated !
class ViTCrossEntropyLoss(AbstractLoss):
    def loss(self, x: torch.tensor):
        return self.xent(self.model(self.up(x)), self.label)

    def __init__(self, model: torch.nn.Module, label: torch.tensor, image_size: int = 384, coefficient: float = 1.):
        super().__init__(coefficient)
        self.model = model
        self.label = label
        self.xent = torch.nn.CrossEntropyLoss()
        self.up = torch.nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False).cuda()


class BatchAugment(AbstractLoss):
    def loss(self, x: torch.tensor):
        if self.aug is not None:
            x = self.aug(x)
        return self.other(x)

    def __init__(self, other: AbstractLoss, aug: torch.tensor = None):
        super().__init__(coefficient=1.0)
        self.other = other
        self.aug = aug


# ===================== From OLD Projects =====================

class NetworkPass(AbstractLoss):
    def __init__(self, model: torch.nn.Module):
        super().__init__(coefficient=0.0)
        self.model = model

    def loss(self, x: torch.tensor):
        self.model(x)
        return torch.tensor(0)


class BatchNorm1stLayer(AbstractLoss):
    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.hook.get_layer(self.layer)

    def reset(self) -> torch.tensor:
        return self.hook.reset()

    def __init__(self, bn_hook: TimedHookHolder, layer: int = 0, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.hook = bn_hook
        self.layer = layer


class ActivationNorm(AbstractLoss):
    def loss(self, x: torch.tensor):
        return - self.hook.get_layer(self.layer)

    def __init__(self, activation_hook: TimedHookHolder, layer: int, coefficient: float = 1.):
        super().__init__(coefficient)
        self.hook = activation_hook
        self.layer = layer

    def reset(self) -> torch.tensor:
        return self.hook.reset()


class ColorDistribution(AbstractLoss):
    def loss(self, x: torch.tensor):
        return self.color_loss(x)

    def __init__(self, normalizer: Normalizer, coefficient: float = 1.):
        super().__init__(coefficient)
        self.color_loss = AbstractColorDistribution(normalizer)
