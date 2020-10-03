import argparse
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger as tb

from dataset import RandomDataModule
from module import Image2ImageModule
from net import SampleModel


def train_args(parent_parser):
    parser = argparse.ArgumentParser(
        parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--batch_size", '-bs', type=int, default=4)
    parser.add_argument(
        "--epochs", type=int, default=4)
    parser.add_argument(
        "--experiment_name", "-exn", type=str,
        default=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))

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


def get_module():
    model = SampleModel(3, 3)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-4)
    scheduler = sched.CosineAnnealingWarmRestarts(
        optimizer, 150)
    criterion = nn.MSELoss()
    return Image2ImageModule(model, optimizer, scheduler, criterion)


def train(args, parser):
    args, parser = train_args(parser)
    tb_logger = tb("..", "experiments", version=args.experiment_name)
    trainer = Trainer(
        gpus=args.gpu,
        logger=tb_logger,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer.fit(get_module(), datamodule=RandomDataModule(args))
