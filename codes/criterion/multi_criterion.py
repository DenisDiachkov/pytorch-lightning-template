import torch
import utils


class MultiCriterion(torch.nn.Module):
    def __init__(self, criterions, criterion_weights: list = None, reduce: str = 'mean'):
        super().__init__()
        self.criterions = []
        for criterion in criterions:
            if isinstance(criterion, dict):
                self.criterions.append(utils.get_instance(criterion.criterion, criterion.criterion_params))
            elif isinstance(criterion, str):
                self.criterions.append(utils.get_instance(criterion))
            elif isinstance(criterion, torch.nn.Module):
                self.criterions.append(criterion)
        self.criterion_weights = criterion_weights if criterion_weights is not None else [1] * len(self.criterions)
        self.reduce = reduce
    
    def forward(self, x, y):
        losses = []
        for i, criterion in enumerate(self.criterions):
            losses.append(criterion(x, y) * self.criterion_weights[i])
        if self.reduce == 'mean':
            return sum(losses) / len(losses)
        elif self.reduce == 'sum':
            return sum(losses)
        elif self.reduce is None:
            return losses
        else:
            raise ValueError(f'Invalid reduce: {self.reduce}')