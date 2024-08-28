"""
This module contains the test function.
"""

from typing import cast

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

from .dataset.datamodule import DataModule
from .utils import get_instance
from .utils.typing import TestConfig


def test(cfg: TestConfig):
    """
    This function tests the model.
    """

    trainer = Trainer(**cfg["trainer_params"])

    # Load the model
    module = cast(
        LightningModule,
        get_instance(cfg["lightning_module"], cfg["lightning_module_params"]),
    )
    # Load the data
    datamodule = DataModule("test", **cfg["datamodule_params"])

    # Test the model
    trainer = pl.Trainer(**cfg["trainer_params"])
    trainer.test(module, datamodule=datamodule)
