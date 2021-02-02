from typing import Sequence

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms as tf
from torchvision.datasets import MNIST, ImageFolder


class DatasetLoader:
    @staticmethod
    def mnist(
            directory: str = "tmp",
            train_flag: bool = True,
            transform: tf.transforms.Compose = None,
            indices: Sequence[int] = None):
        """MNIST

        Args:
            directory (str, optional): [description]. Defaults to "tmp".
            train_flag (bool, optional): [description]. Defaults to True.
            transform (tf.transforms.Compose, optional): [description]. Defaults to None.
            indices (Sequence[int], optional): [description]. Defaults to None.

        Returns:
            dataset: MNIST dataset
        """
        dataset = MNIST(root=directory, train=train_flag, download=True, transform=transform)
        if indices is not None:
            dataset = Subset(dataset, indices)
        return dataset

    @staticmethod
    def image_folder(dirctory: str, transform: tf.transforms.Compose = None):
        """自作画像クラス分類データセット生成

        Args:
            dirctory (str): ディレクトリパス
            transform (tf.transforms.Compose, optional): 画像変換チェーン. Defaults to None.

        Returns:
            dataset: 自作画像クラス分類データセット

        Example:
            /path/to/
            ├─ class1: クラス1の画像があるディレクトリ
            │   ├─ a.jpg
            │   ├─ b.jpg
            │   └─ c.jpg
            ├─ class2: クラス2の画像があるディレクトリ
            │   ├─ a.jpg
            │   ├─ b.jpg
            │   └─ c.jpg
            └─ class3: クラス3の画像があるディレクトリ
                 ├─ a.jpg
                 ├─ b.jpg
                 └─ c.jpg
        """
        return ImageFolder(dirctory, transform=transform)

    @staticmethod
    def make_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
