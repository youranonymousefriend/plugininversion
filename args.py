import argparse
import os
import sys
from typing import Any

import toml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class Args:
    instance = None
    instance_arg = None

    def __init__(self):
        self.args = self._get_parser().parse_args()
        self.args.toml = self.args.toml or os.path.join('configs', '{}.toml'.format(str(sys.argv[0])[:-3]))
        if os.path.exists(self.args.toml):
            self.update()
        else:
            print('Trying to load config file {} but it doesnt exist'.format(self.args.toml))

    def update(self):
        from_file = toml.load(self.args.toml)
        for k, v in from_file.items():
            if not getattr(self.args, k):
                setattr(self.args, k, v)

    @staticmethod
    def _get_parser():
        parser = argparse.ArgumentParser(description='Real Visualization')
        parser.add_argument('-t', '--toml', type=str, help='Path to toml file that contains the default configs')
        parser.add_argument('-x', '--experiment', type=str, help='Experiment name')
        parser.add_argument('-l', '--layer', type=int, default=0, help='# Layer')
        parser.add_argument('-f', '--feature', type=int, default=0, help='# Feature')
        parser.add_argument('-y', '--target', type=int, default=0, help='# Feature')

        parser.add_argument('-m', '--method', type=str, default=None,
                            help='Inversion method, differs for every experiment')
        parser.add_argument('--seed', type=int, default=6247423, help='Random seed')

        parser.add_argument('-a', '--a', type=float, default=0, help='Variable "a" for development')
        parser.add_argument('-b', '--b', type=float, default=0, help='Variable "a" for development')

        parser.add_argument('-n', '--network', type=int, default=0, help='Max 100, get Network')
        return parser

    def get_args(self):
        return self.args

    def get_name(self):
        return '_'.join(['{}{}'.format(k[0], v) for k, v in vars(self.args).items() if v is not None])

    @staticmethod
    def get_instance() -> (argparse.Namespace, Any):
        Args.instance = Args.instance or Args()
        return Args.instance
