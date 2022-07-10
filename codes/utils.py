import os
import random

import numpy as np
import torch
import yaml


def set_device(cfg: dict):
    if cfg['device'] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        cfg['gpu'] = None
    elif cfg['device'][:3] == 'gpu':
        cfg['gpu'] = list(map(int, cfg['device'][4:].split(',')))


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
