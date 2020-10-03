import random

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class RandomImageDataset(Dataset):
    def __init__(self, length, width, height, channels, seed=None):
        self.length = length
        self.width = width
        self.height = height
        self.channels = channels
        self.gen = torch.Generator()
        if seed is None:
            self.seed = random.randint(0, 100000000)
        else:
            self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.gen.manual_seed(idx + self.seed)
        x = torch.rand(
            (self.channels, self.width, self.height, ), generator=self.gen)
        self.gen.manual_seed(idx + 2*self.seed)
        y = torch.rand(
            (self.channels, self.width, self.height, ), generator=self.gen)

        return x, y


class RandomDataModule(LightningDataModule):
    def __init__(self, options):
        super().__init__()
        self.opt = options
        self.num_workers = self.opt.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RandomImageDataset(
                self.opt.length, self.opt.width, self.opt.height,
                self.opt.channels, seed=1)
            self.valid_dataset = RandomImageDataset(
                self.opt.length, self.opt.width, self.opt.height,
                self.opt.channels, seed=2)

        if stage == 'test' or stage is None:
            self.test_dataset = RandomImageDataset(
                self.opt.length, self.opt.width, self.opt.height,
                self.opt.channels, seed=2)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.opt.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.opt.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.opt.batch_size,
            num_workers=self.num_workers
        )
