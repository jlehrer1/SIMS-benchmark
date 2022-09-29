import os
import pathlib
import sys
import anndata as an
import torch
import argparse

from os.path import join, dirname, abspath

sys.path.append(join(dirname(abspath(__file__)), ".."))

from typing import *
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from scsims import *
from torchmetrics.functional import *
from networking import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        required=False,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-4,
        required=False,
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        required=False,
    )

    device = "cuda:0" if torch.cuda.is_available() else None

    args = parser.parse_args()
    lr, weight_decay, name, test = args.lr, args.weight_decay, args.name, args.test

    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, "..", "data", "benchmark")

    print("Making data folder")
    os.makedirs(data_path, exist_ok=True)

    for file in ["mouse_labels_clean.csv", "mouse_clipped.h5ad"]:
        print(f"Downloading {file}")

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join("jlehrer", "mouse_benchmark", file),
                file_name=join(data_path, file),
            )

    class_label = "subclass_label"
    module = DataModule(
        datafiles=[join(data_path, "mouse_clipped.h5ad")],
        labelfiles=[join(data_path, "mouse_labels_clean.csv")],
        class_label=class_label,
        sep=",",
        batch_size=256,
        index_col="cell",
        num_workers=32,
        deterministic=True,
        normalize=True,
        assume_numeric_label=False,
    )

    wandb_logger = WandbLogger(
        project=f"Mouse Cortical Model",
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    upload_callback = UploadCallback(path="checkpoints", desc="mouse_cortical_model")

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
    )

    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
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

    if not test:
        module.prepare_data()
        module.setup()

        model = SIMSClassifier(
            input_dim=module.num_features,
            output_dim=module.num_labels,
            weights=module.weights,
        )

        trainer.fit(model, datamodule=module)
        trainer.test(model, datamodule=module)
    else:
        raise NotImplementedError("No checkpoints downloaded yet")
