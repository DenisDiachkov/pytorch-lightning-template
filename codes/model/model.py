from pytorch_lightning import LightningModule
import torch


class Image2ImageModule(LightningModule):
    def __init__(self,
                 in_features: int = 3,
                 out_features: int = 3,
                 **kwargs
                 ):
        # init superclass
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()

        self.conv1 = torch.nn.Conv2d(
            self.hparams.in_features,
            8, 3, padding=1)

        self.conv2 = torch.nn.Conv2d(
            8,
            self.hparams.out_features, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = torch.nn.MSELoss()(p, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = torch.nn.MSELoss()(p, y)
        tensorboard_logs = {'train_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
        return [optimizer], [scheduler]
