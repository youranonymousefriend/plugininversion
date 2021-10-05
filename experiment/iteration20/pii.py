from augmentation import Clip, Jitter, Focus, RepeatBatch, Zoom, ColorJitterR
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, CrossEntropyLoss
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    network = args.network
    target, way = args.target, int(args.a)
    model, image_size, batch_size, name = model_library[network]()

    # saver = ExperimentSaver(f'VarNetworkNorm{network}_{way}_{target}', save_id=True, disk_saver=True)
    saver = ExperimentSaver(f'VarNetLRS_{network}_{way}_{target}', save_id=True, disk_saver=True)

    loss = LossArray()
    loss += CrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)

    post = torch.nn.Sequential(Clip())
    image = torch.rand(size=(1, 3, image_size, image_size)).cuda()

    step = image_size // 8
    padding = [0, step // 2, step, 0]
    pad = padding[way]

    print(image_size)
    for i in range(2*step, image_size + 1, step):
        image = new_init(i, last=image, padding=pad)
        seq = [Focus(i, 0), Jitter(), RepeatBatch(batch_size), ColorJitterR(batch_size, True)]
        pre = torch.nn.Sequential(*seq)

        visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=50, lr=.01, steps=400, save_every=400)
        image.data = visualizer(image).data
        print(f'==================={i}================')


if __name__ == '__main__':
    main()
