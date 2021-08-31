import torch
from torch import nn
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy
import os

class BaselineModel(pl.LightningModule):
    def __init__(self, input_channels, n_classes):
        super().__init__()
        self.batch0 = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 64 * 48, 64)
        self.batch2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, n_classes)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
    
    def forward(self, x):
        #print(x.shape)
        out = self.batch0(x)
        out = F.max_pool2d(F.relu(self.conv1(out)), 2)
        #print("pool ",out.shape)
        out = F.max_pool2d(F.relu(self.batch1(self.conv2(out))), 2)
        #print("pool ",out.shape)
        out = out.view(-1, 8 * 64 * 48)
        #print(out.shape)
        out = F.relu(self.batch2(self.fc1(out)))
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)
        return out

    def loss(self, x, y):
        logits = self(x)  # this calls self.forward
        loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log('training loss/epoch', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, y)
        self.log('training acc/epoch', self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid loss/epoch", loss)  # default on val/test is on_epoch only
        self.log('valid acc/epoch', self.valid_acc)
            
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-04)


class BaselineModelBlock(nn.Module):
    def __init__(self):
        pass