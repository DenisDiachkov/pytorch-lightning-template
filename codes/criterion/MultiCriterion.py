import torch


class MultiCriterion(torch.nn.Module):
    def __init__(self, *criterions: torch.nn.Module, weights: list):
        super().__init__()
        self.criterions = criterions

    def forward(self, x, y):
        losses = []
        for criterion in self.criterions:
            losses.append(criterion(x, y))
        return sum(losses)