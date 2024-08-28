"""Example test file for pytest."""

from pathlib import Path
from typing import cast

import pytest

from pytorch_lightning_template.train import train
from pytorch_lightning_template.utils import read_yaml_config
from pytorch_lightning_template.utils.typing import TrainConfig


@pytest.fixture
def config_path() -> Path:
    """Fixture for the path to the configuration file."""

    return Path(__file__).parent / "overfit.yaml"


@pytest.fixture
# pylint: disable=redefined-outer-name
def config(config_path: Path) -> TrainConfig:
    """Fixture for the configuration."""

    return cast(TrainConfig, read_yaml_config(config_path))


# pylint: disable=redefined-outer-name
def test_overfit(config: TrainConfig):
    """Test the overfitting of the model."""
    train(config)
