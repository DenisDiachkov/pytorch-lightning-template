from dataset.data_module import RandomDataModule
from pytorch_lightning import Trainer
from model.model import Image2ImageModule
import pickle2reducer
import multiprocessing as mp

ctx = mp.get_context()
ctx.reducer = pickle2reducer.Pickle2Reducer()


def train(args):
    data_module = RandomDataModule(args)
    module = Image2ImageModule()

    trainer = Trainer()
    trainer.fit(module, datamodule=data_module)
