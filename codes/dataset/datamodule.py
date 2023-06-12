import utils
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np
import pickle


class DataModule(LightningDataModule):
    def __init__(
        self,
        mode: str,
        dataset: str,
        dataset_params: dict = {},
        
        train_dataset_params: dict = {},
        val_dataset_params: dict = {},
        test_dataset_params: dict = {},
        dataloader_params: dict = {},

        train_dataloader_params: dict = {},
        val_dataloader_params: dict = {},
        test_dataloader_params: dict = {},
    ):
        super().__init__()
        if mode == 'train':
            self.data_train = utils.get_instance(dataset, {'mode': 'train'} | dataset_params | train_dataset_params)
            self.data_val = utils.get_instance(dataset, {'mode': 'val'} | dataset_params | val_dataset_params)
        elif mode == 'test':
            self.data_test = utils.get_instance(dataset, {'mode': 'test'} | dataset_params | test_dataset_params)

        self.dataset_test_params = {'mode': 'test'} | dataset_params | test_dataset_params

        self.dataloader_params = dataloader_params
        self.train_dataloader_params = train_dataloader_params
        self.val_dataloader_params = val_dataloader_params
        self.test_dataloader_params = test_dataloader_params
        
    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            **self.dataloader_params | self.train_dataloader_params,
            worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            **self.dataloader_params | self.val_dataloader_params,
            worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            **self.dataloader_params | self.test_dataloader_params,
            worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )
