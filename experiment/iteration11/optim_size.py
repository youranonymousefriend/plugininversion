from augmentation import Clip, Jitter, Focus, ColorJitter, RepeatBatch, Zoom, ColorJitterR
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit, get_default_model, get_default_model_not_norm
from saver import ExperimentSaver
from utils import exp_starter_pack
from datasets import inv_dataset
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    target, dims = args.target, int(args.a)
    constructor, batch_size, name, image_size = get_default_model, 32, 'DiffSize', 224

    saver = ExperimentSaver(f'{name}{dims}_{target}', save_id=True, disk_saver=True)
    model = constructor().cuda()

    loss = LossArray()
    loss += CrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)

    post = torch.nn.Sequential(Clip())
    image = torch.rand(size=(1, 3, image_size // dims, image_size // dims)).cuda()

    for i in range(32, image_size + 1, 32):
        seq = [Zoom(out_size=image_size), Focus(i, 0), Jitter(), RepeatBatch(batch_size),
               ColorJitterR(batch_size, True)]
        pre = torch.nn.Sequential(*seq)

        visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=.1, steps=400, save_every=100)
        image.data = visualizer(image).data


if __name__ == '__main__':
    main()
