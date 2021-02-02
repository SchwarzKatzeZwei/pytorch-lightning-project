import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from cnn_finetune import make_model

from dataset_loader import DatasetLoader
from models.cnnnet import CNNNet
from transforms import Transforms


class PLModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # self.net = CNNNet()
        self.net = make_model('resnet18', num_classes=2, pretrained=True)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y_hat = self(x)
        y_hat = self.net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        # log step metric
        self.log('train_acc_step', self.accuracy(y_hat, y))
        return loss

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return torch.optim.Adam(self.parameters())

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())


def main():
    transform = Transforms.compose()
    # train_dataset = DatasetLoader.mnist(transform=transform)
    train_dataset = DatasetLoader.image_folder("tmp/train", transform=transform)
    train_loader = DatasetLoader.make_loader(train_dataset, batch_size=10, num_workers=4)

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
