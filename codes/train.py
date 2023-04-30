import os
from email import utils

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import utils
from dataset.datamodule import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger as tb


def train(cfg: dict):
    if not os.path.exists(cfg.experiment_path):
        os.makedirs(cfg.experiment_path)
    tb_logger = tb(".", "experiments", version=cfg.experiment_name)
    callbacks = [
        utils.get_obj(callback.callback)(
            **callback.callback_params | ({"dirpath": cfg.experiment_path} if callback.callback.endswith("ModelCheckpoint") else {})
        ) 
        for callback in cfg.trainer_callbacks
    ]
    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        **cfg.trainer_params
    )
    trainer.fit(
        utils.get_obj(cfg.lightning_module)(cfg.lightning_module_params),
        datamodule=DataModule(cfg.mode, **cfg.datamodule_params), 
        ckpt_path=cfg.resume_path
    )
