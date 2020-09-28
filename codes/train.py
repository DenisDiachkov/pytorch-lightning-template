import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger as tb

from dataset.data_module import RandomDataModule
from model.model import Image2ImageModule
from model.net import SampleModel


def get_module():
    model = SampleModel(3, 3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = sched.CosineAnnealingWarmRestarts(optimizer, 150)
    criterion = nn.MSELoss()
    return Image2ImageModule(model, optimizer, scheduler, criterion)


def train(args):
    tb_logger = tb("..", "experiments", version=args.experiment_name)
    trainer = Trainer(
        gpus=args.gpu,
        logger=tb_logger,
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer.fit(get_module(), datamodule=RandomDataModule(args))
