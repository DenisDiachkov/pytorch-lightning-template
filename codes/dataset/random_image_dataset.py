import torch
from torch.utils.data import Dataset
import random


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
        x = torch.random(
            (self.width, self.height, self.channels), generator=self.gen)
        self.gen.manual_seed(idx + 2*self.seed)
        y = torch.random(
            (self.width, self.height, self.channels), generator=self.gen)

        return x, y
