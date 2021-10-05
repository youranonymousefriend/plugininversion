from augmentation import Clip, Jitter, Focus, ColorJitter
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch
from datasets.inits import inv_dataset


def main():
    exp_name, args, _ = exp_starter_pack()
    target = args.target
    saver = ExperimentSaver(f'Color{target}', save_id=True, disk_saver=True)
    batch_size = 1

    # model = get_default_robust().cuda()
    model = get_robust_normal().cuda()

    # model3 = get_default_vit().cuda()

    loss = LossArray()
    loss += CrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)
    # loss += CrossEntropyLoss(model2, torch.tensor([target] * batch_size).long().cuda(), coefficient=.5)
    # loss += ViTCrossEntropyLoss(model3, torch.tensor([target]).long().cuda(), coefficient=.5)
    loss += TotalVariation(coefficient=0.0005)

    pre = torch.nn.Sequential(Clip())
    mx_size = 256
    # image = torch.rand(size=(1, 3, mx_size, mx_size)).cuda()
    image = inv_dataset[target] 
    for i in range(mx_size, mx_size + 1, 32):
        post = torch.nn.Sequential(Focus(i, 0), Jitter(),)

        visualizer = ImageNetVisualizer(loss, saver, post, pre, print_every=10, lr=0.1, steps=300, save_every=100)
        image.data = visualizer(image).data
    post = torch.nn.Sequential(Focus(i, 0), Jitter())
    visualizer = ImageNetVisualizer(loss, saver, post, pre, print_every=10, lr=0.1, steps=300, save_every=100)
    image.data = visualizer(image).data


if __name__ == '__main__':
    main()
