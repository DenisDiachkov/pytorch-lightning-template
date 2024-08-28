"""
Module for the LightningModule
"""

from typing import Any, Literal

import torch
from pytorch_lightning import LightningModule

from .utils import utils


# pylint: disable=too-many-ancestors
class BaseModule(LightningModule):
    """
    Base class for the LightningModule.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model: str,
        model_params: dict,
        optimizer: str | None = None,
        optimizer_params: dict | None = None,
        scheduler: str | None = None,
        scheduler_params: dict | None = None,
        criterion: str | None = None,
        criterion_params: dict | None = None,
        log_metrics_every_n_steps: int = 1,
        visualization_every_n_steps: int = 30,
    ):
        super().__init__()
        self.model: torch.nn.Module = utils.get_instance(model, model_params)
        if optimizer is not None:
            self.optimizer = utils.get_instance(
                optimizer,
                {"params": self.model.parameters()} | (optimizer_params or {}),
            )
        if scheduler is not None:
            self.scheduler = utils.get_instance(
                scheduler, {"optimizer": self.optimizer} | (scheduler_params or {})
            )
        if criterion is not None:
            self.criterion = utils.get_instance(criterion, criterion_params or {})

        self.log_metrics_every_n_steps = log_metrics_every_n_steps
        self.visualization_every_n_steps = visualization_every_n_steps

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(kwargs["batch"]["input"])

    # pylint: disable=unused-argument
    def calc_loss(self, *args: Any, **kwargs: Any) -> Any:
        """
        This function calculates the loss.
        """
        return self.criterion(kwargs["batch"]["output"], kwargs["batch"]["target"])

    def log_metrics(self, *args, **kwargs) -> None:
        """
        This function logs the metrics.
        """

    def visualization(self, *args, **kwargs) -> None:
        """
        This function visualizes the results.
        """

    def log_all(
        self,
        mode: Literal["train", "val", "test"],
        batch: Any,
        loss: Any,
        batch_idx: int,
        **kwargs,
    ) -> None:
        """
        This function logs all the metrics.
        """

        # Log the loss
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=True)

        # Log the learning rate
        if hasattr(self, "scheduler"):
            self.log(
                "lr", self.scheduler.get_last_lr()[0], prog_bar=True, sync_dist=True
            )

        # Log the metrics
        if (batch_idx + 1) % self.log_metrics_every_n_steps == 0:
            self.log_metrics(batch)

        # Visualize the results
        if (batch_idx + 1) % self.visualization_every_n_steps == 0:
            self.visualization(batch)

    def save_test_results(self, batch: Any, batch_idx: int) -> None:
        """
        This function saves the test results.
        """

    # pylint: disable=arguments-differ
    def training_step(self, batch, batch_idx: int):
        batch["output"] = self(batch=batch, batch_idx=batch_idx)
        loss = self.calc_loss(batch=batch).mean()
        self.log_all("train", batch, loss, batch_idx)
        return loss

    # pylint: disable=arguments-differ, unused-argument
    def validation_step(self, batch, batch_idx):
        batch["output"] = self(batch=batch, batch_idx=batch_idx)
        loss = self.calc_loss(batch=batch).mean()
        self.log_all("val", batch, loss, batch_idx)

    # pylint: disable=arguments-differ, unused-argument
    def test_step(self, batch, batch_idx):
        batch["output"] = self(batch=batch, batch_idx=batch_idx)
        loss = self.calc_loss(batch=batch).mean()
        self.log_all("test", batch, loss, batch_idx)
        self.save_test_results(batch, batch_idx)

    def configure_optimizers(self) -> Any:
        if self.optimizer is None:
            return None
        conf = {
            "optimizer": self.optimizer,
        }
        if hasattr(self, "scheduler"):
            conf["scheduler"] = self.scheduler
        return conf
