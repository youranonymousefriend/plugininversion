import torch
import torch.nn as nn


class BasicHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.base_hook_fn)
        self.activations = None

    def close(self):
        self.hook.remove()

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t
        # x = output_t
        x = x[0][0] if isinstance(x[0], tuple) else x[0]
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError


class ItemIterator:
    @property
    def iterator_item(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.iterator_item)

    def __getitem__(self, item):
        return self.iterator_item[item]

    def __len__(self):
        return len(self.iterator_item)
