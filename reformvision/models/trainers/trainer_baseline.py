import math
import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from reformvision.models.baseline_model import BaselineModel
from reformvision.data_modules.utils import train_val_split

if __name__ == "__main__":
    train, val, classes = train_val_split(os.getcwd() + '/data/spectra_data_examples/data')
    train_dataloader = DataLoader(train, batch_size=64, num_workers=4)
    val_dataloader = DataLoader(val, batch_size=64, num_workers=4)

    base_model = BaselineModel(3, len(classes))

    seed_everything(42)
    # wandb_logger = WandbLogger()
    # trainer = pl.Trainer(logger=wandb_logger, deterministic=True, max_epochs=10)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(base_model, train_dataloader, val_dataloader)
    