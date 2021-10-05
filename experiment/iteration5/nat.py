from augmentation import Clip, Jitter, Focus, ColorJitter, RepeatBatch, Zoom, ColorJitterR
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit, get_default_model, get_default_model_not_norm
from saver import ExperimentSaver
from utils import exp_starter_pack
from datasets import inv_dataset
import torch

experiments = {'basic': {'network': get_default_model_not_norm, 'batch_size': 32, 'old_name': 'UnNormalNatUnJitter'},
               'jitter': {'network': get_default_model_not_norm, 'batch_size': 32, 'old_name': 'UnNormalNat'},
               'normal': {'network': get_default_model, 'batch_size': 32, 'old_name': 'NatUnJitter'},
               'normal_jitter': {'network': get_default_model, 'batch_size': 32, 'old_name': 'Nat'},
               'normal_jitter1': {'network': get_default_model, 'batch_size': 32, 'old_name': 'NatJitterOnce'},
               'normal_real_jitter': {'network': get_default_model, 'batch_size': 32, 'old_name': 'RealJitter'},
               'normal_real_jitter_many': {'network': get_default_model, 'batch_size': 32, 'old_name': 'RealJitterMany'}}


def main():
    exp_name, args, _ = exp_starter_pack()
    target = args.target

    values = experiments[args.method]
    constructor, batch_size, name = values['network'], values['batch_size'], values['old_name']
    saver = ExperimentSaver(f'{name}{target}', save_id=True, disk_saver=True)
    mx_size = 256

    model = constructor().cuda()

    loss = LossArray()
    loss += CrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)

    post2 = torch.nn.Sequential(Clip())
    shrink = 2
    image = torch.rand(size=(1, 3, mx_size // shrink, mx_size // shrink)).cuda()
    last = None

    for i in range(32, mx_size + 1, 32):
        seq = [Zoom(out_size=mx_size), Focus(i, 0), Jitter()]
        if args.method in ['jitter', 'normal_jitter']:
            seq += [RepeatBatch(batch_size), ColorJitter(batch_size, True)]
        elif args.method == 'normal_jitter1':
            seq += [RepeatBatch(batch_size), ColorJitter(batch_size, False)]
        elif args.method == 'normal_real_jitter':
            seq += [RepeatBatch(batch_size), ColorJitterR(batch_size, False)]
        elif args.method == 'normal_real_jitter_many':
            seq += [RepeatBatch(batch_size), ColorJitterR(batch_size, True)]
        pre = torch.nn.Sequential(*seq)
        # pre = torch.nn.Sequential(Zoom(out_size=mx_size), Focus(cur_size, 0), Jitter(), RepeatBatch(batch_size),
        #                           ColorJitter(batch_size))

        visualizer = ImageNetVisualizer(loss, saver, pre, post2, print_every=10, lr=.1, steps=400, save_every=100)
        image.data = visualizer(image).data
        last = image


if __name__ == '__main__':
    main()
