import torch
import utils
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataset.datamodule import DataModule


def get_new_test_func(test_func):
    def new_test_func(x, batch_idx):
        output = test_func(x, batch_idx)
    return new_test_func


def test(cfg: dict):
    tester = Trainer(
        logger=False,
        **cfg.trainer_params,
    )
    if cfg.checkpoint_path.endswith('.ckpt'):
        train_cfg = torch.load(cfg.checkpoint_path)['cfg']
        module = utils.get_instance(train_cfg.lightning_module, train_cfg.lightning_module_params)
    else:
        raise NotImplementedError
    module.test_step = get_new_test_func(module.test_step)
    datamodule=DataModule(cfg.mode, **cfg.datamodule_params)
    tester.test(
        module,
        datamodule=datamodule,
        ckpt_path=cfg.checkpoint_path
    )