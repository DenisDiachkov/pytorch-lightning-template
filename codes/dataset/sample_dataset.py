import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(
        self, 
        in_features: int = 100,
        out_features: int = 100,
        num_samples: int = 1000,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        '''
        return a sample
        '''
        x = torch.randn(self.in_features)
        y = torch.tanh(x)
        return x, y
