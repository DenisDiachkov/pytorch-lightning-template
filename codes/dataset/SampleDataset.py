import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(
        self, 
        mode:str, 
        cfg: dict
    ):
        self.cfg = cfg
        self.in_features = cfg['in_features']
        self.out_features = cfg['out_features']
        self.set_mode(mode)

    def set_mode(self, mode: str):
        '''
        set paths, num_samples, etc
        '''
        if mode == 'train':
            self.num_samples = self.cfg['num_train_samples']
        elif mode == 'val':
            self.num_samples = self.cfg['num_val_samples']
        elif mode == 'test':
            self.num_samples = self.cfg['num_test_samples']
            
        pass 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        '''
        return a sample
        '''
        x = torch.randn(self.in_features)
        y = torch.tanh(x)
        return x, y
