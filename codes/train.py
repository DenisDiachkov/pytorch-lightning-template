from dataset.data_module import RandomDataModule
from pytorch_lightning import Trainer
from model.model import Image2ImageModule
from pytorch_lightning.loggers import TensorBoardLogger as tb


def train(args):
    data_module = RandomDataModule(args)
    module = Image2ImageModule()

    tb_logger = tb(".", "experiments", version=args.experiment_name)

    trainer = Trainer(
        gpus=args.gpu,
        logger=tb_logger,
        num_sanity_val_steps=1,
        deterministic=True
         #distributed_backend="ddp"
    )
    trainer.fit(module, datamodule=data_module)
