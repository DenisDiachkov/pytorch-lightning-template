import utils
from dataset.datamodule import DataModule
from pytorch_lightning import Trainer


def test(cfg: dict):
    tester = Trainer(
        logger=False,
        **cfg.trainer_params,
    )
    tester.test(
        utils.get_obj(cfg.lightning_module)(cfg.lightning_module_params),
        datamodule=DataModule(cfg.mode, **cfg.datamodule_params), 
        ckpt_path=cfg.checkpoint_path
    )
