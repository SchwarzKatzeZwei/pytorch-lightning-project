import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import MNIST

from models.cnnnet import CNNNet
from transforms import Transforms


class CNNModel(pl.LightningModule):

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
    train_set = MNIST(root='dataset', train=True, download=True, transform=transform)
    train_set = Subset(train_set, list(range(0, 1000)))
    train_loader = DataLoader(train_set, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # init model
    model = CNNModel()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
