import os
import random

import numpy as np
import torch
import yaml



def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_config(cfg: dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(cfg, f)
