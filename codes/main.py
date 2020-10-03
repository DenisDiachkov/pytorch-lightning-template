import argparse
import multiprocessing as mp
import warnings
from test import test

import utils
from train import train


def base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=['train', 'test'], default='train')
    parser.add_argument(
        "--gpu", type=str)
    parser.add_argument(
        "--cpu", action="store_true")
    parser.add_argument(
        "--num_workers", "--jobs", "-j",
        type=int, choices=range(mp.cpu_count()+1), default=mp.cpu_count())
    parser.add_argument("--Wall", action="store_true")

    args, _ = parser.parse_known_args()
    utils.set_device(args)
    return args, parser


def main():
    args, parser = base_args()
    if args.Wall:
        warnings.simplefilter("error")
    if args.mode == "train":
        train(args, parser)
    elif args.mode == "test":
        test(args, parser)


if __name__ == "__main__":
    main()
