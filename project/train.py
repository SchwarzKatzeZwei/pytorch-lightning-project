import re

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from dataset_loader import DatasetLoader
from models.cnnnet import CNNNet
from transforms import Transforms


class PLModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = CNNNet()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


def main():
    transform = Transforms.compose()
    train_dataset = DatasetLoader.mnist(transform=transform, indices=list(range(0, 1000)))
    train_loader = DatasetLoader.make_loader(train_dataset, batch_size=100, num_workers=4)

    # init model
    model = PLModule()

    # make pytorch lightning trainer
    trainer_args = {
        "max_epochs": 5
    }
    if torch.cuda.is_available():
        trainer_args["gpus"] = -1
    trainer = pl.Trainer(**trainer_args)

    # train
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
