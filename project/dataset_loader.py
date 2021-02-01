from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms as tf
from torchvision.datasets import MNIST
from typing import Sequence


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
    def make_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
