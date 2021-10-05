from augmentation import Clip, Jitter, Focus, ColorJitter, RepeatBatch
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    target = args.target
    saver = ExperimentSaver(exp_name, save_id=True, disk_saver=True)
    batch_size = 1

    # model = get_default_robust().cuda()
    model = get_robust_normal().cuda()

    loss = LossArray()
    loss += CrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)
    loss += TotalVariation(coefficient=0.000005)

    pre = torch.nn.Sequential(Clip())
    mx_size = 256
    image = torch.rand(size=(1, 3, mx_size, mx_size)).cuda()
    for i in range(32, mx_size + 1, 32):
        post = torch.nn.Sequential(Focus(i, 0), Jitter(), RepeatBatch(batch_size), ColorJitter(batch_size))

        visualizer = ImageNetVisualizer(loss, saver, post, pre, print_every=10, lr=0.1, steps=300, save_every=100)
        image.data = visualizer(image).data
    for j in range(1):
        post = torch.nn.Sequential(Focus(mx_size, 0), Jitter(), RepeatBatch(batch_size), ColorJitter(batch_size))
        visualizer = ImageNetVisualizer(loss, saver, post, pre, print_every=10, lr=0.1, steps=300, save_every=100)
        image.data = visualizer(image).data


if __name__ == '__main__':
    main()
