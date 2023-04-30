import argparse
import os
import warnings
from test import test

import sconf
import utils
from train import train


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/train/SampleCFG.yaml', type=str
    )
    args, left_argv = parser.parse_known_args()
    cfg = sconf.Config(args.cfg)
    cfg.argv_update(left_argv)
    return cfg


def save_cfg(cfg: dict):
    cfg.experiment_path = os.path.join(".", "experiments", cfg.experiment_name)
    if not os.path.exists(cfg.experiment_path):
        os.makedirs(cfg.experiment_path)
    with open(os.path.join(cfg.experiment_path, "config.yaml"), "w") as f:
        f.write(cfg.dumps(modified_color=None))


def main():
    cfg = parse_cfg()
    if cfg.trainer_params.deterministic:
        utils.fix_seed(cfg['seed'])
    if cfg.wall:
        warnings.simplefilter("error")
    if cfg.mode == "train":
        save_cfg(cfg)
        train(cfg)
    elif cfg.mode == "test":
        test(cfg)
    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":
    main()
