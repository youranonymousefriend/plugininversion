from torchvision.models.inception import inception_v3

from model.base import ModelLibrary, TorchVisionModel


class Inception(TorchVisionModel):
    pass


inceptions = ModelLibrary([
    Inception(inception_v3, 38),
])
