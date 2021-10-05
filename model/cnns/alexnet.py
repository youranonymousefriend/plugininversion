from torchvision.models.alexnet import alexnet

from model.base import ModelLibrary, TorchVisionModel


class AlexNet(TorchVisionModel):
    pass


alexnets = ModelLibrary([
    AlexNet(alexnet, 33),
])
