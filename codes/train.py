import os
from email import utils

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import utils
import pytorch_lightning as pl
from dataset.datamodule import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


# Crutch for PyTorch Lightning to stop logging useless metrics
class _Wandblogger(WandbLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        metrics.pop('epoch', None)
        metrics.pop('global_step', None)
        super().log_metrics(metrics=metrics, step=step)


class LogCodeAndConfigCallback(pl.Callback):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        trainer.logger.experiment.log_code(
            root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        trainer.logger.experiment.config.update({**self.cfg})



def train(cfg: dict):
    if not os.path.exists(cfg.experiment_path):
        os.makedirs(cfg.experiment_path)
    logger = _Wandblogger(**cfg.logger_params)
    callbacks = [
        utils.get_obj(callback.callback)(
            **callback.callback_params | ({"dirpath": cfg.experiment_path} if callback.callback.endswith("ModelCheckpoint") else {})
        ) 
        for callback in cfg.trainer_callbacks
    ] + [LogCodeAndConfigCallback(cfg)]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer_params
    )
    trainer.fit(
        utils.get_obj(cfg.lightning_module)(cfg.lightning_module_params),
        datamodule=DataModule(cfg.mode, **cfg.datamodule_params), 
        ckpt_path=cfg.resume_path
    )
