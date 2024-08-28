"""Main.py"""

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import cast

from . import __version__
from .test import test  # type: ignore
from .train import train
from .utils import utils
from .utils.typing.config import TestConfig, TrainConfig


def parse_cfg() -> TrainConfig | TestConfig:
    """
    This function parses the configuration file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg/train.yaml", type=str)
    args, _ = parser.parse_known_args()
    cfg = utils.read_yaml_config(args.cfg)
    return TrainConfig(**cfg) if cfg["mode"] == "train" else TestConfig(**cfg)


def save_cfg(cfg: TrainConfig):
    """
    This function saves the configuration file.
    """

    experiment_path = (
        Path(".")
        / "experiments"
        / cfg["experiment_name"]
        / (
            cfg["logger_params"]["version"]
            if cfg["logger_params"]["version"] is not None
            else ""
        )
    )
    utils.dump_yaml_config(cfg, experiment_path / "config.yaml")
    cfg["experiment_path"] = experiment_path


def set_loglevel(level: str):
    """
    This function sets the log level.
    """

    level = level.lower()
    if level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    elif level == "info":
        logging.basicConfig(level=logging.INFO)
    elif level == "warning":
        logging.basicConfig(level=logging.WARNING)
    elif level == "error":
        logging.basicConfig(level=logging.ERROR)
    elif level == "critical":
        logging.basicConfig(level=logging.CRITICAL)
    else:
        raise ValueError("Invalid log level")


def main():
    """
    This function is the main function.
    """

    logging.info("pytorch-lightning-template: %s", __version__)

    cfg = parse_cfg()
    set_loglevel(cfg["loglevel"])
    if cfg["environ_vars"] is not None:
        for k, v in cfg["environ_vars"].items():
            os.environ[k] = str(v)
    if cfg["trainer_params"]["deterministic"]:
        utils.fix_seed(cfg["seed"])
    if cfg["wall"]:
        warnings.simplefilter("error")
    if cfg["mode"] == "train":
        cfg = cast(TrainConfig, cfg)
        save_cfg(cfg)
        train(cfg)
    elif cfg["mode"] == "test":
        cfg = cast(TestConfig, cfg)
        test(cfg)
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    main()
