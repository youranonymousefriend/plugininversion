import timm

from datasets import image_net
from ..augmented import AugmentedModel
from ..base import InvertModel
import torch


class TransformerModel(InvertModel):
    def __call__(self, normalize: bool = True):
        model, image_size, batch_size, name = super(TransformerModel, self).__call__(normalize)
        up = torch.nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False).cuda()
        model = AugmentedModel(model, up).cuda()
        return model, image_size, batch_size, name


class TimmModel(TransformerModel):
    options = []
    def get_size_based_on_name(self, name:str) -> int:
        return 384 if '384' in name else 224

        

    def __init__(self, o: int, batch_size: int):
        def get_from_timm(option: int = 0) -> torch.nn.Module:
            """ Hats off to: https://github.com/facebookresearch/DeiT """
            return timm.create_model(self.options[option], pretrained=True)

        image_size = self.get_size_based_on_name(self.options[o])
        super(TimmModel, self).__init__(f'{self.__class__.__name__}_{o}_{self.options[o]}', get_from_timm, {'option': o},
                                        image_size, batch_size, image_net.normalizer)
