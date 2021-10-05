import torch
import torchvision
from torch.utils.data import DataLoader

from utils import exp_starter_pack
from datasets import image_net
from tqdm import tqdm
import pdb


def main():
    exp_name, args, _ = exp_starter_pack()
    target = args.target
    real_size = 64

    up = torch.nn.Upsample(size=(real_size, real_size), mode='bilinear', align_corners=False).cuda()

    data = image_net.train()
    good_indices = [i for i in tqdm(range(len(data.samples))) if data.samples[i][1] == target]
    images = []
    for i in tqdm(good_indices):
        x, y = data[i]
        images.append(x)
    images = torch.stack(images).cuda()
    images = up(images)
    torchvision.utils.save_image(images, f'desktop/ImageNet{target}.png', nrow=(224 * 8) // real_size)


if __name__ == '__main__':
    main()
