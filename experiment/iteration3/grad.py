import json
import random

from torch.utils.data import Dataset, DataLoader

from loss import LossArray, TotalVariation, CrossEntropyLoss
from augmentation import ColorJitter
from model import get_default_robust, get_robust_normal
from utils import exp_starter_pack
import torch
import numpy as np
import pdb
from datasets import image_net
from tqdm import tqdm


class GridDataset(Dataset):
    @staticmethod
    def orthogonal(a: torch.tensor, b: torch.tensor) -> torch.tensor:
        b = b.view(-1)
        a = a.view(-1)
        a = a / a.norm()
        a = a - (a @ b) * b
        a = a / a.norm()
        return a

    def random_orthogonal(self, count: int = 3, shape: torch.Size = (1, 3, 224, 224)) -> torch.tensor:
        out = []
        for i in range(count):
            cur = torch.rand(np.prod(shape))
            cur = cur / cur.norm()
            for other in out:
                cur = self.orthogonal(cur, other)
            out.append(cur.view(shape))
        return torch.cat(out)

    @staticmethod
    def get_grid(n_dims: int = 1, start: float = -1, end: float = 1, step: float = 9):
        cur = torch.arange(start, end, step)
        count = cur.shape[-1]
        print(count)
        output = torch.zeros(size=[count] * n_dims + [n_dims])
        for i in range(n_dims):
            cur = cur.view([1] * i + [-1] + [1] * (n_dims - i - 1))
            output.select(-1, i).add_(cur)
        return output

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> torch.tensor:
        return ((self.axis * self.flat_grid[index]).sum(dim=0) + self.center).detach().clone()

    def __init__(self, center: torch.tensor = None, grid_size: float = 0.1, grid_unit: float = 0.01,
                 shape: torch.Size = (1, 3, 224, 224)):
        self.axis = self.random_orthogonal(3, shape)
        self.grid = self.get_grid(3, -grid_size, +grid_size + 0.0001, grid_unit)
        self.flat_grid = self.grid.view(-1, 3, 1, 1, 1)
        self.flat_axis = self.axis.view(3, -1).cuda()
        self.size = self.grid.view(-1, 3).size(0)
        if center is None:
            center = torch.rand(shape).squeeze(0)
        self.center = center

    def __call__(self, grad: torch.tensor, normalize: bool = True) -> torch.tensor:
        grad = grad.view((grad.size(0), -1)).T
        normalized = grad / grad.norm(dim=0).view(1, -1) if normalize else grad
        output = (self.flat_axis @ normalized).T
        return output


def main():
    exp_name, args, _ = exp_starter_pack()
    target = args.target
    batch_size = 1

    x, y = image_net.eval()[random.randint(0, 1000 * 100)]
    print(y)

    dataset = GridDataset(center=x, grid_size=10, grid_unit=3)
    loader = DataLoader(dataset, batch_size, shuffle=False)
    label = torch.tensor([target] * batch_size).long().cuda()
    off_manifold = get_loss_dirs(get_default_robust().cuda(), label, loader, dataset, pre_aug=ColorJitter(1))
    with open('js6/off.js', 'w') as file:
        json.dump(off_manifold.detach().cpu().numpy().tolist(), file)
    # dataset.flat_grid = 0.1 * dataset.flat_grid; 
    # dataset.grid = 0.1 * dataset.grid; 
    on_manifold = get_loss_dirs(get_robust_normal().cuda(), label, loader, dataset)
    with open('js6/on.js', 'w') as file:
        json.dump(on_manifold.detach().cpu().numpy().tolist(), file)
    pdb.set_trace()


def get_loss_dirs(model, label, loader, dataset, pre_aug=None) -> torch.tensor:
    loss = LossArray()
    loss += CrossEntropyLoss(model, label, coefficient=1.)
    loss += TotalVariation(coefficient=0.0005)

    grads = []
    for x in tqdm(loader):
        x = x.cuda()
        x.requires_grad_()
        xx = x if pre_aug is None else pre_aug(x)
        loss(xx).backward()
        grads.append(dataset(x.grad.detach().clone().cuda()))
    grads = torch.cat(grads)
    real_size = int(np.cbrt(grads.shape[0]))
    return grads.view(real_size, real_size, real_size, 3)


if __name__ == '__main__':
    main()
