"""
This module contains the main training loop for the model.
"""

import os
from typing import Any, Dict, cast

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import TorchCheckpointIO  # type: ignore
from pytorch_lightning.utilities import rank_zero_only  # type: ignore

from .dataset.datamodule import DataModule
from .utils import get_instance
from .utils.typing import TrainConfig  # type: ignore


# Crutch for PyTorch Lightning to stop logging useless metrics
class _Wandblogger(WandbLogger):
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step=None) -> None:  # type: ignore[override]
        metrics.pop("epoch", None)
        metrics.pop("global_step", None)
        super().log_metrics(metrics=metrics, step=step)


class LogCodeAndConfigCallback(pl.Callback):
    """
    Callback to log the code and config to wandb.
    """

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: pl.LightningModule):
        trainer.logger.experiment.log_code(  # type: ignore
            root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        trainer.logger.experiment.config.update(  # type: ignore
            {**self.config}, allow_val_change=True
        )


class CheckpointIO(TorchCheckpointIO):
    """
    This class is a crutch to save the config in the checkpoint.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        checkpoint["cfg"] = self.cfg
        super().save_checkpoint(checkpoint, path, storage_options=storage_options)


def train(cfg: TrainConfig):
    """
    This function trains the model.
    """

    if not os.path.exists(cfg["experiment_path"]):
        os.makedirs(cfg["experiment_path"])
    logger = _Wandblogger(**cfg["logger_params"]) if not cfg["no_logging"] else None
    callbacks = [
        get_instance(
            callback["callback"],
            callback["callback_params"]
            | (
                {"dirpath": cfg["experiment_path"]}
                if callback["callback"].endswith("ModelCheckpoint")
                else {}
            ),
        )
        for callback in cfg["trainer_callbacks"]
    ]
    if not cfg["no_logging"]:
        callbacks.append(LogCodeAndConfigCallback(cfg))
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,  # type: ignore
        plugins=[CheckpointIO(cfg)],
        **cfg["trainer_params"],
    )
    datamodule = DataModule("train", **cfg["datamodule_params"])
    trainer.fit(
        cast(
            LightningModule,
            get_instance(cfg["lightning_module"], cfg["lightning_module_params"]),
        ),
        datamodule=datamodule,
        ckpt_path=cfg["resume_path"],
    )
