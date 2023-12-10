from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import models, transforms, datasets

from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.nn as nn


class MyModel(pl.LightningModule):
    def __init__(self, lr=0.001, step_size=20, gamma=0.1):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_features, 4)
        self.logged_metrics = []

        self.criterion = torch.nn.CrossEntropyLoss()

        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [lr_scheduler]
    
    def on_validation_epoch_end(self):
            # Save the logged metrics at the end of each epoch
            metrics = self.trainer.callback_metrics.copy()

            self.logged_metrics.append(metrics)
