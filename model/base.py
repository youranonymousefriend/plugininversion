import torch

from datasets import Normalizer, image_net
from hooks.base import ItemIterator
from model.augmented import AugmentedModel


class InvertModel(ItemIterator):
    @property
    def iterator_item(self):
        return [self]

    def __init__(self, name: str, constructor, constructor_args: dict, image_size: int, batch_size: int,
                 normalizer: Normalizer):
        self.name = name
        self.constructor = constructor
        self.constructor_args = constructor_args
        self.image_size = image_size
        self.batch_size = batch_size
        self.normalizer = normalizer

    def __call__(self, normalize: bool = True) -> (torch.nn.Module, int, int, str):
        model = self.constructor(**self.constructor_args)
        if normalize:
            model = AugmentedModel(model, self.normalizer)
        model.eval()
        return model.cuda(), self.image_size, self.batch_size, self.name


class TorchVisionModel(InvertModel):
    def __init__(self, constructor, batch_size: int):
        name = f'{self.__class__.__name__}_{constructor.__name__}'
        constructor_args = {'pretrained': True}
        image_size = 224
        normalizer = image_net.normalizer
        super().__init__(name, constructor, constructor_args, image_size, batch_size, normalizer)


class ModelLibrary(ItemIterator):
    @property
    def iterator_item(self):
        return self.models

    def __init__(self, other_models: list):
        self.models = [m for l in other_models for m in l]
