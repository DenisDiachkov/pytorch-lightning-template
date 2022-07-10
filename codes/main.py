import argparse
import multiprocessing as mp
import warnings
from test import test

import yaml

import utils
from train import resume, train


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/SampleCFG.yaml', type=str
    )
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    for key, val in cfg.items():
        if type(val) is bool:
            parser.add_argument(
                f'--{key}',
                default=val, action=argparse.BooleanOptionalAction
            )
        else: 
            parser.add_argument(
                f'--{key}',
                default=val, type=type(val)
            )
    args = parser.parse_args()
    cfg = vars(args)
    
    print(cfg, type(val))
    return cfg


def main():
    cfg = parse_cfg()
    utils.set_device(cfg)
    if cfg['deterministic']:
        utils.fix_seed(cfg['seed'])
    if cfg['wall']:
        warnings.simplefilter("error")
    if cfg['mode'] == "train":
        train(cfg)
    elif cfg['mode'] == "test":
        test(cfg)
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    main()
