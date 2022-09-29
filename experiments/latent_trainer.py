import sys
from os.path import join, dirname, abspath

sys.path.append(join(dirname(abspath(__file__)), ".."))
sys.path.append(join(dirname(abspath(__file__)), "..", "networking.py"))

import pandas as pd
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from scsims.autoencoder import *
from torchmetrics.functional import *
from scsims import *
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from typing import *
import sys
import anndata as an
import torch
import argparse
from networking import *
import torch.nn as nn

import gc


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=16,
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        default="allen/human.h5ad",  # just a random default
    )

    args = parser.parse_args()
    name, batch_size, file = args.name, args.batch_size, args.file
    print("BATCH SIZE IS", batch_size)
    # Download training data
    print(f"Downloading {file}")
    download(
        remote_name=join("mostajo_group/single_cell", file),
        file_name=file.split("/")[-1],
    )
    data = an.read_h5ad(file.split("/")[-1])

    def train_val_dataset(dataset, val_split=0.25):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    dataset = AEDataset(data.X)
    train, val = train_val_dataset(dataset)

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=32)
    valloader = DataLoader(val, batch_size=batch_size, num_workers=32)

    train_compresser = AutoEncoder(
        data_shape=data.shape[1],
        encoder_layers=nn.Sequential(
            nn.BatchNorm1d(data.shape[1]),
            nn.Linear(data.shape[1], 10000),
            nn.ReLU(),
            nn.BatchNorm1d(10000),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
        ),
        decoder_layers=nn.Sequential(
            nn.BatchNorm1d(5000),
            nn.Linear(5000, 10000),
            nn.ReLU(),
            nn.BatchNorm1d(10000),
            nn.Linear(10000, data.shape[1]),
            nn.ReLU(),
            nn.BatchNorm1d(data.shape[1]),
        ),
        optim_params={
            "optimizer": torch.optim.Adam,
            "lr": 0.01,
            "weight_decay": 0.000,
        },
    )

    wandb_logger = WandbLogger(
        project=f"Autoencoder allen data",
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    upload_callback = UploadCallback(path="checkpoints", desc="allen_autoencoder")

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
    )

    trainer = pl.Trainer(
        gpus=(-1 if torch.cuda.is_available() else 0),
        auto_lr_find=False,
        logger=wandb_logger,
        max_epochs=500,
        gradient_clip_val=0.5,
        callbacks=[
            lr_callback,
            upload_callback,
            early_stopping_callback,
        ],
    )

    trainer.fit(train_compresser, trainloader, valloader)
