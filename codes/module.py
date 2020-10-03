from pytorch_lightning import LightningModule
import torch


class Image2ImageModule(LightningModule):
    def __init__(self, model, optimizer, scheduler, criterion):
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = self.criterion(p, y)
        
        tensorboard_logs = {
            'train_loss': loss
        }
        return {
            'loss': loss, 
            'log': tensorboard_logs
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = self.criterion(p, y)
        
        tensorboard_logs = {
            'train_loss': loss
        }
        return {
            'val_loss': loss, 
            'log': tensorboard_logs
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'Avg loss': avg_loss
        }
        return {
            'val_loss': avg_loss,
            'log': tensorboard_logs
        }

    def configure_optimizers(self):
        if self.optimizer is None:
            return None
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]
        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]
        return self.optimizer, self.scheduler
