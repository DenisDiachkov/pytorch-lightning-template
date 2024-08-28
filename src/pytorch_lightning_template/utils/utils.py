"""
This module contains utility functions.
"""

import importlib
import os
import random
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from ruamel.yaml import YAML

from .. import __package__ as GLOBAL_PACKAGE_NAME


def fix_seed(seed: int):
    """
    This function fixes the seed for reproducibility.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_cls(obj_path: str):
    """
    This function returns the class specified by obj_path.
    """

    if obj_path is None:
        return None

    if obj_path.startswith("."):
        obj_path = cast(str, GLOBAL_PACKAGE_NAME) + obj_path

    module_name, obj_name = obj_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def get_instance(obj_path: str, params: Any) -> Any:
    """
    This function returns an instance of the class
    specified by obj_path with the parameters specified by params.
    """

    cls = get_cls(obj_path)
    if cls is None:
        return None
    return cls(**params)


def read_yaml_config(file_path: Path) -> Mapping[str, Any]:
    """
    This function reads a yaml file and returns a dictionary.
    """

    yaml = YAML(typ="safe", pure=True)
    with open(file_path, "r", encoding="utf-8") as f:
        config_dict = yaml.load(f)
    return config_dict


def dump_yaml_config(
    config_dict: Mapping[str, Any], file_path: Path, makedirs: bool = True
):
    """
    This function writes a dictionary to a yaml file.
    """

    if makedirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML(typ="safe", pure=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
