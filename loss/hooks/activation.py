import pdb

import torch
from torch import nn as nn
from hooks.base import BasicHook
from datetime import datetime


class AbstractMiniBatchActivation(BasicHook):
    def __init__(self, module: nn.Module, seed: int = 0, targets: list = None):
        super().__init__(module)
        self.activations = []
        self.seed = seed
        self.targets = targets

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

    def reset(self):
        if self.activations is not None:
            for t, v in self.activations:
                del v
            del self.activations
        self.activations = []

    def set_seed(self, seed: int):
        self.seed = seed

    def set_target(self, target: list):
        self.targets = target


class MiniBatchFeatureActivationHook(AbstractMiniBatchActivation):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.seed:]
        diagonal = torch.arange(min(input_t.size()[:2]))
        feats = input_t[diagonal, diagonal]
        self.activations.append((datetime.now(), feats.norm(p=2, dim=(1, 2)).mean()))


class TargetFeatureActivationHook(AbstractMiniBatchActivation):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.seed:]
        diagonal = torch.arange(min(input_t.size()[:2]))
        feats = input_t[diagonal, self.targets]
        self.activations.append((datetime.now(), feats.norm(p=2, dim=(1, 2)).mean()))


class MiniBatchContrastiveActivationHook(AbstractMiniBatchActivation):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.seed:]
        size = min(input_t.size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        value = size * feats.norm(p=2, dim=(1, 2)).mean() - input_t[diagonal].norm(p=2, dim=(2, 3)).mean()
        self.activations.append((datetime.now(), value))


class ViTActivationCLS(AbstractMiniBatchActivation):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.seed:]
        size = min(input_t.size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats[:, 0].mean() * feats.size(-1)
        self.activations.append((datetime.now(), feats))


class ViTActivationMean(AbstractMiniBatchActivation):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.seed:]
        size = min(input_t.size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats.norm(p=2, dim=-1).mean() * 10 * 10
        self.activations.append((datetime.now(), feats))
