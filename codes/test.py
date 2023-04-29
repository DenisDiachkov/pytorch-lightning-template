import torch.nn as nn
from pytorch_lightning import Trainer

from criterion.MultiCriterion import MultiCriterion
from dataset.datamodule import DataModule
from dataset.SampleDataset import SampleDataset
from models.SampleModel import SampleModel
from module import BaseModule


def get_criterion(cfg: dict):
    if cfg['criterion'] == 'MSELoss':
        cls = nn.MSELoss
    elif cfg['criterion'] == 'CrossEntropyLoss':
        cls = nn.CrossEntropyLoss
    elif cfg['criterion'] == 'L1Loss':
        cls = nn.L1Loss
    elif cfg['criterion'] == 'MultiCriterion':
        cls = MultiCriterion
        criterions = []
        for criterion in cfg['criterions']:
            criterions.append(get_criterion(criterion))
        return cls(*criterions, weights=cfg['criterion_weights'])
    else:
        raise ValueError(f'Unknown criterion {cfg["criterion"]}')
    return cls(**cfg['criterion_params'])


def get_module(cfg: dict):
    model = SampleModel(cfg['in_features'], cfg['out_features'])
    print(cfg)
    criterion = get_criterion(cfg)
    return BaseModule(model, None, None, criterion)


def test(cfg: dict):
    tester = Trainer(
        devices=cfg['gpu'],
        deterministic=cfg['deterministic'],
        logger=False
    )
    tester.test(get_module(cfg), datamodule=DataModule(cfg, SampleDataset), ckpt_path=cfg["pretrained_model"])
