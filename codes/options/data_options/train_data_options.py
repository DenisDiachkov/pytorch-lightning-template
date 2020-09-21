import argparse
from argparse import ArgumentError


def data_train_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--length", '-n', type=int, default=100)
    parser.add_argument(
        "--width", '-W', type=int, default=4)
    parser.add_argument(
        "--height", '-H', type=int, default=4)
    parser.add_argument(
        "--channels", '-c', type=int, default=3)

    args, _ = parser.parse_known_args()
    return args, parser
