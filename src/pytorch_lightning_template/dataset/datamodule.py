"""
DataModule class for PyTorch Lightning
"""

from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..utils import utils
from ..utils.typing.config import DataLoadersParams


# pylint: disable=too-many-instance-attributes, too-many-arguments
class DataModule(LightningDataModule):
    """
    Generic DataModule class for PyTorch Lightning

    Args:
        mode (str): Mode of the DataModule (train or test)
        dataset (str): Dataset class
        dataset_params (dict): Parameters for the dataset
        train_dataset_params (dict): Parameters for the train dataset
        val_dataset_params (dict): Parameters for the validation dataset
        test_dataset_params (dict): Parameters for the test dataset
        dataloader_params (DataLoadersParams): Parameters for the dataloader
        train_dataloader_params (DataLoadersParams): Parameters for the train dataloader
        val_dataloader_params (DataLoadersParams): Parameters for the validation dataloader
        test_dataloader_params (DataLoadersParams): Parameters for the test dataloader
    """

    def __init__(
        self,
        mode: str,
        dataset: str,
        dataset_params: Optional[dict] = None,
        train_dataset_params: Optional[dict] = None,
        val_dataset_params: Optional[dict] = None,
        test_dataset_params: Optional[dict] = None,
        dataloader_params: Optional[DataLoadersParams] = None,
        train_dataloader_params: Optional[DataLoadersParams] = None,
        val_dataloader_params: Optional[DataLoadersParams] = None,
        test_dataloader_params: Optional[DataLoadersParams] = None,
    ):
        dataset_params = dataset_params or {}
        train_dataset_params = train_dataset_params or {}
        val_dataset_params = val_dataset_params or {}
        test_dataset_params = test_dataset_params or {}
        dataloader_params = dataloader_params or {}
        train_dataloader_params = train_dataloader_params or {}
        val_dataloader_params = val_dataloader_params or {}
        test_dataloader_params = test_dataloader_params or {}

        super().__init__()
        if mode == "train":
            self.data_train = utils.get_instance(
                dataset, {"mode": "train"} | dataset_params | train_dataset_params
            )
            self.data_val = utils.get_instance(
                dataset, {"mode": "val"} | dataset_params | val_dataset_params
            )
        elif mode == "test":
            self.data_test = utils.get_instance(
                dataset, {"mode": "test"} | dataset_params | test_dataset_params
            )

        self.dataset_test_params = (
            {"mode": "test"} | dataset_params | test_dataset_params
        )

        self.dataloader_params = dataloader_params
        self.train_dataloader_params = train_dataloader_params
        self.val_dataloader_params = val_dataloader_params
        self.test_dataloader_params = test_dataloader_params

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            **self.dataloader_params | self.train_dataloader_params,
            worker_init_fn=lambda worker_id: np.random.seed(
                # Makes sure that the random seed is different for each worker
                np.random.get_state()[1][0]  # type: ignore
                + worker_id
            ),
            collate_fn=(
                self.data_train.collate_fn
                if hasattr(self.data_train, "collate_fn")
                else None
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            **self.dataloader_params | self.val_dataloader_params,
            worker_init_fn=lambda worker_id: np.random.seed(
                # Makes sure that the random seed is different for each worker
                np.random.get_state()[1][0]  # type: ignore
                + worker_id
            ),
            collate_fn=(
                self.data_val.collate_fn
                if hasattr(self.data_val, "collate_fn")
                else None
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            **self.dataloader_params | self.test_dataloader_params,
            worker_init_fn=lambda worker_id: np.random.seed(
                # Makes sure that the random seed is different for each worker
                np.random.get_state()[1][0]  # type: ignore
                + worker_id
            ),
            collate_fn=(
                self.data_test.collate_fn
                if hasattr(self.data_test, "collate_fn")
                else None
            ),
        )
