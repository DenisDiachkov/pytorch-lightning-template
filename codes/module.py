import torch
from pytorch_lightning import LightningModule


class BaseModule(LightningModule):
    def __init__(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.Module,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        return self.criterion(self(x=x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x=x), y)
        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x=x), y)
        self.log('test_loss', loss, prog_bar=True)
    
    def configure_optimizers(self):
        if self.optimizer is None:
            return None
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]
        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]
        return self.optimizer, self.scheduler
