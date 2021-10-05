from augmentation import Clip, Jitter, Focus, ColorJitter, RepeatBatch, Zoom, ColorJitterR
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit, get_default_model, get_default_model_not_norm
from saver import ExperimentSaver
from utils import exp_starter_pack
from datasets import inv_dataset
import torch


def new_init(size: int, batch_size: int = 1, last: torch.nn = None, padding: int = -1) -> torch.nn:
    output = torch.rand(size=(batch_size, 3, size, size)).cuda()
    if last is not None:
        big_size = size if padding == -1 else size - padding
        up = torch.nn.Upsample(size=(big_size, big_size), mode='bilinear', align_corners=False).cuda()
        scaled = up(last)
        cx = (output.size(-1) - big_size) // 2
        output[:, :, cx:cx + big_size, cx:cx + big_size] = scaled
    output = output.detach().clone()
    output.requires_grad_()
    return output


def main():
    exp_name, args, _ = exp_starter_pack()
    target, way = args.target, int(args.a)
    constructor, batch_size, name, image_size = get_default_vit, 1, 'Vit_noaug_con', 384

    saver = ExperimentSaver(f'{name}{way}_{target}', save_id=True, disk_saver=True)
    model = constructor().cuda()

    loss = LossArray()
    loss += ViTCrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)

    post = torch.nn.Sequential(Clip())
    if way != 3: 
        image = None
    else:
        image = torch.rand(size=(1, 3, image_size, image_size)).cuda()

    step = 48
    padding = [0, step // 2, step, 0]
    pad = padding[way]

    for i in range(step, image_size + 1, step):
        if way != 3: 
            image = new_init(i, last=image, padding=pad)
            seq = [Focus(i, 0), Zoom(out_size=image_size), Jitter()]
        else: 
            image = new_init(image_size, last=image, padding=pad)
            seq = [Focus(image_size, 0), Zoom(out_size=image_size), Jitter()]
        pre = torch.nn.Sequential(*seq)

        visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=.1, steps=400, save_every=200)
        image.data = visualizer(image).data


if __name__ == '__main__':
    main()
