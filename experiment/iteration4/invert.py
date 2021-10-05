from augmentation import Clip, Jitter, Focus, ColorJitter, RepeatBatch, Zoom
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit
from saver import ExperimentSaver
from utils import exp_starter_pack
from datasets import inv_dataset
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    target = args.target
    saver = ExperimentSaver(exp_name, save_id=True, disk_saver=True)
    batch_size = 32
    mx_size = 256

    model = get_default_robust().cuda()

    loss = LossArray()
    loss += CrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)
    loss += TotalVariation(size=mx_size, coefficient=0.000005)

    post2 = torch.nn.Sequential(Clip())
    image = torch.rand(size=(1, 3, mx_size, mx_size)).cuda()
    last = None

    for i in range(2*32, mx_size + 1, 32):
        pre = torch.nn.Sequential(Focus(i, 0), Jitter(), RepeatBatch(batch_size), ColorJitter(batch_size))
        # pre = torch.nn.Sequential(Zoom(out_size=mx_size), Focus(cur_size, 0), Jitter(), RepeatBatch(batch_size),
        #                           ColorJitter(batch_size))

        visualizer = ImageNetVisualizer(loss, saver, pre, post2, print_every=10, lr=0.1, steps=300, save_every=100)
        image.data = visualizer(image).data
        last = image


if __name__ == '__main__':
    main()
