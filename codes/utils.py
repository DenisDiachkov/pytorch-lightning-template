import importlib
import random

import numpy as np
import torch


def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_obj(obj_path: str):
    module_name, obj_name = obj_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
