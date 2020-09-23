import argparse
from .data_options.train_data_options import data_train_args
from datetime import datetime


def train_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--batch_size", '-bs', type=int, default=4)
    parser.add_argument(
        "--epochs", type=int, default=4)
    parser.add_argument(
        "--experiment_name", "-exn", type=str,
        default=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    args, parser = data_train_args(parser)
    
    return args
