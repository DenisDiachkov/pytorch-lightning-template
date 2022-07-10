import argparse
import os
from datetime import datetime
from email import utils

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger as tb

import utils
from criterion.MultiCriterion import MultiCriterion
from dataset.datamodule import DataModule
from dataset.SampleDataset import SampleDataset
from models.SampleModel import SampleModel
from module import BaseModule


def get_optimizer(model, cfg):
    if cfg['optimizer'] == 'Adam':
        cls = optim.Adam
    elif cfg['optimizer'] == 'SGD':
        cls =  optim.SGD
    else:
        raise ValueError(f'Unknown optimizer {cfg["optimizer"]}')
    return cls(model.parameters(), **cfg['optimizer_params'])


def get_scheduler(optimizer, cfg):
    if cfg['scheduler'] == 'CosineAnnealingWarmRestarts':
        cls = sched.CosineAnnealingWarmRestarts
    elif cfg['scheduler'] == 'StepLR':
        cls = sched.StepLR
    else:
        raise ValueError(f'Unknown scheduler {cfg["scheduler"]}')
    return cls(optimizer, **cfg['scheduler_params'])


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
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    criterion = get_criterion(cfg)
    return BaseModule(model, optimizer, scheduler, criterion)


def train(cfg: dict):
    experiment_path = os.path.join(".", "experiments", cfg['experiment_name'])
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    tb_logger = tb(".", "experiments", version=cfg['experiment_name'])
    utils.save_config(cfg, os.path.join(experiment_path, "config.yaml"))

    # Checkpointing
    checkpoint_callback_lightning = ModelCheckpoint(
        experiment_path,
        save_top_k=2,
        save_last=True,
        monitor="val_loss",
        mode="min",
        verbose=True,
    )
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg['patience'],
        verbose=True,
        mode="min",
        min_delta=cfg['min_delta'],
    )            

    trainer = Trainer(
        gpus=cfg['gpu'],
        logger=tb_logger,
        num_sanity_val_steps=1,
        deterministic=cfg['deterministic'],
        max_epochs=cfg['epochs'],
        callbacks=[early_stop_callback],
        checkpoint_callback=checkpoint_callback_lightning,
    )
    trainer.fit(get_module(cfg), datamodule=DataModule(cfg, SampleDataset))


def resume(cfg: dict):
    experiment_path = os.path.join(".", "experiments", cfg['experiment_name'])
    trainer = Trainer.load_from_checkpoint(
        os.path.join(experiment_path, "checkpoints", "best.ckpt")
    )
    trainer.resume_from_checkpoint(
        os.path.join(experiment_path, "checkpoints", "last.ckpt")
    )
    trainer.fit(get_module(cfg), datamodule=DataModule(cfg, SampleDataset))
