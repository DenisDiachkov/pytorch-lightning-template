from pytorch_lightning import LightningModule
import torch


class SampleModel(LightningModule):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 3,
        **kwargs
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_features, 8, 3, padding=1)

        self.conv2 = torch.nn.Conv2d(
            8, out_features, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))