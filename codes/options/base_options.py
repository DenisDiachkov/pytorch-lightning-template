import argparse
import multiprocessing as mp
from argparse import ArgumentError
import os


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
    set_device(args)
    return args, parser


def set_device(args):
    if args.cpu is not None:
        if args.gpu is not None:
            raise ArgumentError("Can't use CPU and GPU at the same time")
    elif args.gpu is None:
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]
