import random

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DataModule(LightningDataModule):
    def __init__(
        self,
        cfg: dict, 
        data_cls: type = Dataset,
    ):
        super().__init__()
        self.cfg = cfg
        self.data_cls = data_cls
        if cfg['mode'] == 'train':
            self.data_train = self.data_cls("train", self.cfg)
            self.data_val = self.data_cls("val", self.cfg)
        if cfg['mode'] == 'test':
            self.data_test = self.data_cls("test", self.cfg)
        
    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, num_workers=self.cfg['num_workers']
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.cfg['batch_size'], 
            shuffle=True, num_workers=self.cfg['num_workers']
        )
    
    def test_dataloader(self):
        return DataLoader(self.data_test, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, num_workers=self.cfg['num_workers']
        )
