from typing import Any
import torch
import utils
from multipledispatch import dispatch
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod


class BaseModule(LightningModule, ABC):
    def __init__(
            self, 
            model:str, model_params: dict, 
            optimizer:str=None, optimizer_params: dict = None, 
            scheduler:str=None, scheduler_params: dict = None, 
            criterion:str=None, criterion_params: dict = None
        ):
        super().__init__()
        self.model = utils.get_instance(model, model_params)
        self.optimizer = utils.get_instance(optimizer, {'params':self.model.parameters()} | optimizer_params)
        self.scheduler = utils.get_instance(scheduler, {'optimizer':self.optimizer} | scheduler_params) 
        self.criterion = utils.get_instance(criterion, criterion_params)
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def calc_loss(self, batch):
        pass

    def training_step(self, batch, batch_idx: int):
        batch['output'] = self(batch)
        loss = self.calc_loss(batch).mean()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch['output'] = self(batch)
        loss = self.calc_loss(batch).mean()
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        batch['output'] = self(batch)
        loss = self.calc_loss(batch).mean()
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        if self.optimizer is None:
            return None
        return {
            'optimizer': self.optimizer, 
            'scheduler': self.scheduler
        }


class DummyModule(BaseModule):
    def __init__(
            self, 
            model:str, model_params: dict, 
            optimizer:str=None, optimizer_params: dict = None, 
            scheduler:str=None, scheduler_params: dict = None, 
            criterion:str=None, criterion_params: dict = None
        ):
        super().__init__(
            model, model_params, 
            optimizer, optimizer_params, 
            scheduler, scheduler_params, 
            criterion, criterion_params
        )
    
    def forward(self, x):
        return self.model(x)
    
    def calc_loss(self, batch):
        return self.criterion(batch['output'], batch['target'])