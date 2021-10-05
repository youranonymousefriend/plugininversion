import torch
from pytorch_pretrained_vit import ViT
from torch import nn as nn

from datasets import weird_image_net
from .augmented import AugmentedModel, BNModel
from datasets.imagenet import image_net
from loss import BaseFakeBN
from torchvision.models import resnet18, resnet50

from .base import ModelLibrary
from .cnns import convolutionals
from .transformers import vits

model_library = ModelLibrary([convolutionals, vits])


def _parallel_cuda(func):
    def to_parallel() -> nn.Module:
        model = func()
        model = nn.DataParallel(model.cuda())
        return model

    return to_parallel


def get_default_model() -> nn.Module:
    model = resnet50(pretrained=True)
    model = AugmentedModel(model, image_net.normalizer)
    model.eval()
    return model.cuda()


def get_default_model_not_norm() -> nn.Module:
    model = resnet50(pretrained=True)
    model.eval()
    return model.cuda()


def get_default_vit() -> nn.Module:
    model = ViT('B_16_imagenet1k', pretrained=True)
    model = AugmentedModel(model, weird_image_net.normalizer)
    model.eval()
    return model.cuda()


def get_vit_no_aug() -> nn.Module:
    model = ViT('B_16_imagenet1k', pretrained=True)
    model.eval()
    return model.cuda()


def get_default_robust() -> nn.Module:
    model = nn.DataParallel(resnet50().cuda())
    checkpoint = torch.load('./checkpoints/free/free.pt')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_robust_normal() -> nn.Module:
    model = get_default_robust()
    model = AugmentedModel(model, image_net.normalizer)
    return model


def default_bn():
    model = resnet18(pretrained=True)
    bn = BNModel(model.conv1, nn.BatchNorm2d(model.conv1.out_channels)).cuda()
    bn = BaseFakeBN(bn, image_net, 'imagenet_0.pth').cuda()
    bn.eval()
