"""

Fashion-MNIST dataset.

"""

from typing import Any, Dict, List, Optional, cast

import albumentations as A
import numpy as np
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from .. import utils


class FashionMNIST(Dataset):
    """
    Fashion-MNIST dataset.

    Args:
        mode (str): Mode of the dataset.
        root (str): Root directory of the dataset.
    """

    def __init__(
        self, mode: str, root: str, albumentations_transform: Optional[A.Compose] = None
    ):
        self.root = root
        self.mode = mode
        self.dataset = torchvision.datasets.FashionMNIST(
            root=root,
            train=mode == "train",
            transform=None,
            download=True,
        )

        # Initialize albumentations transforms
        if isinstance(albumentations_transform, dict):
            compose_list: List[A.TransformType] = []
            for transform, params in albumentations_transform.items():
                compose_list.append(utils.get_instance(transform, params))
            albumentations_transform = A.Compose(compose_list)

        self.albumentations_transform = A.Compose(
            [
                cast(A.Compose, albumentations_transform) or A.NoOp(),
                ToTensorV2(),
            ],
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "input": self.albumentations_transform(
                image=np.array(self.dataset[index][0], np.float32)
            )["image"],
            "target": self.dataset[index][1],
        }

    def __len__(self) -> int:
        return len(self.dataset)
