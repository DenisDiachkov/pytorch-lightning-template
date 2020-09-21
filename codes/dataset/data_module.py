from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .random_image_dataset import RandomImageDataset


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
