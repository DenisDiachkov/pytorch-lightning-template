import importlib
import random
import os

import numpy as np
import torch


def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_cls(obj_path: str):
    if obj_path is None:
        return None
    module_name, obj_name = obj_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def get_instance(obj_path: str, params: dict = {}):
    cls = get_cls(obj_path)
    if cls is None:
        return None
    return cls(**params)
