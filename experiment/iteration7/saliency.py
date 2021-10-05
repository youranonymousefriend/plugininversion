from torch.utils.data import DataLoader
import pdb

from augmentation import Clip, Jitter, Focus, ColorJitter, RepeatBatch, Zoom, ColorJitterR
from inversion import ImageNetVisualizer
from loss import LossArray, TotalVariation, CrossEntropyLoss, ViTCrossEntropyLoss
from model import get_default_robust, get_robust_normal, get_default_vit, get_default_model, get_default_model_not_norm
from saver import ExperimentSaver
from utils import exp_starter_pack
from datasets import inv_dataset
from saliency_tools import VanillaBackProp, GuidedBackProp
from tqdm import tqdm
import torch
from datasets import image_net

experiments = {
    'basic': {'network': get_default_model_not_norm, 'batch_size': 1, 'old_name': 'SaliencyUnNormalNatUnJitter'},
    'jitter': {'network': get_default_model_not_norm, 'batch_size': 32, 'old_name': 'SaliencyUnNormalNat'},
    'normal': {'network': get_default_model, 'batch_size': 1, 'old_name': 'SaliencyNatUnJitter'},
    'normal_jitter': {'network': get_default_model, 'batch_size': 32, 'old_name': 'SaliencyNat'},
    'normal_jitter1': {'network': get_default_model, 'batch_size': 32, 'old_name': 'SaliencyNatJitterOnce'},
    'normal_real_jitter': {'network': get_default_model, 'batch_size': 32, 'old_name': 'SaliencyRealJitter'},
    'normal_real_jitter_many': {'network': get_default_model, 'batch_size': 32,
                                'old_name': 'SaliencyRealJitterMany'}}


def main():
    exp_name, args, _ = exp_starter_pack()
    data = image_net.eval()

    values = experiments[args.method]
    constructor, batch_size, name = values['network'], values['batch_size'], values['old_name']
    saver = ExperimentSaver(f'{name}', save_id=False, disk_saver=True)
    model = constructor().cuda()

    seq = []
    if args.method in ['jitter', 'normal_jitter']:
        seq += [RepeatBatch(batch_size), ColorJitter(batch_size, True)]
    elif args.method == 'normal_jitter1':
        seq += [RepeatBatch(batch_size), ColorJitter(batch_size, False)]
    elif args.method == 'normal_real_jitter':
        seq += [RepeatBatch(batch_size), ColorJitterR(batch_size, False)]
    elif args.method == 'normal_real_jitter_many':
        seq += [RepeatBatch(batch_size), ColorJitterR(batch_size, True)]
    pre = torch.nn.Sequential(*seq)
    network = torch.nn.Sequential(pre, model)

    sal_method = 'guided'
    sal_method = 'vanilla'
    saliency_object = VanillaBackProp(network) if sal_method == 'vanilla' else GuidedBackProp(network)

    for target in tqdm(range(0, 50, 5)):
        image = data[target*50][0].unsqueeze(0).cuda()
        image.requires_grad_()

        # cur_loss = loss(augmented)
        # cur_loss.backward()
        # sal = image.grad.detach().clone()
        sal = saliency_object.generate_gradients(image, target)
        sal = sal.norm(dim=1, p=1)
        sal_min = sal.min()
        sal_max = sal.max()
        sal = (sal - sal_min) / (sal_max - sal_min)

        saver.save(image, f'r_{target}')
        saver.save(sal, f'{sal_method}_{target}')


if __name__ == '__main__':
    main()
