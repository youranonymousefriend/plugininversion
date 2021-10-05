from torchvision.models.googlenet import googlenet

from model.base import ModelLibrary, TorchVisionModel


class GoogleNet(TorchVisionModel):
    pass


googlenets = ModelLibrary([
    GoogleNet(googlenet, 66),
])
