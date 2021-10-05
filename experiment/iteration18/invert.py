from augmentation import Clip, Jitter, Focus, ColorJitter
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    target = args.target
    batch_size = int(args.a)
    steps = 300
    saver = ExperimentSaver(f'BatchSize{target}{batch_size}', save_id=True, disk_saver=True)

    model = get_default_robust().cuda()

    loss = LossArray()
    loss += CrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)
    loss += TotalVariation(coefficient=0.0005)

    pre = torch.nn.Sequential(Clip())
    mx_size = 256
    image = torch.rand(size=(1, 3, mx_size, mx_size)).cuda()
    for i in range(32, mx_size + 1, 32):
        post = torch.nn.Sequential(Focus(i, 0), Jitter(), ColorJitter(batch_size))

        visualizer = ImageNetVisualizer(loss, saver, post, pre, print_every=10, lr=0.1, steps=steps, save_every=steps)
        image.data = visualizer(image).data
    post = torch.nn.Sequential(Focus(i, 0), Jitter(), ColorJitter(batch_size))
    visualizer = ImageNetVisualizer(loss, saver, post, pre, print_every=10, lr=0.1, steps=steps, save_every=steps)
    image.data = visualizer(image).data


if __name__ == '__main__':
    main()
