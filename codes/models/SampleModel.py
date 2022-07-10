import torch
from pytorch_lightning import LightningModule


class SampleModel(LightningModule):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 3,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
