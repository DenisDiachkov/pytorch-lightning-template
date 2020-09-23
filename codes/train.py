from dataset.data_module import RandomDataModule
from pytorch_lightning import Trainer
from model.model import Image2ImageModule
import multiprocessing as mp


def train(args):
    data_module = RandomDataModule(args)
    module = Image2ImageModule()

    trainer = Trainer()
    trainer.fit(module, datamodule=data_module)
