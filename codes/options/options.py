from .base_options import base_args
from .train_options import train_args


def parse_args():
    args, parser = base_args()
    if args.mode == 'train':
        args = train_args(parser)
    else:
        args = test_args(parser)
    return args
