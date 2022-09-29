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
    name, test = args.name, args.test

    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, "..", "data", "benchmark")

    print("Making data folder")
    os.makedirs(data_path, exist_ok=True)

    labels = list_objects("jlehrer/benchmark/human_labels")
    # Download training labels set
    for file in labels:
        print(f"Downloading {file}")

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join(file),
                file_name=join(data_path, file.split("/")[-1]),
            )

    # Download training data
    for file in ["human_labels_clean.csv", "human.h5ad", "combined_organoid_data_for_test.h5ad"]:
        print(f"Downloading {file}")

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join("jlehrer", "human_benchmark", file),
                file_name=join(data_path, file),
            )

    combined_data = an.read_h5ad(join(data_path, "combined_organoid_data_for_test.h5ad"))
    allen = an.read_h5ad(join(data_path, "human.h5ad"))

    combined_genes = [x.upper() for x in combined_data.var["index"].values]
    currgenes = [x.upper() for x in allen.var.index]
    refgenes = list(set(currgenes).intersection(combined_genes))

    module = DataModule(
        datafiles=[join(data_path, "human.h5ad")],
        labelfiles=[join(data_path, "human_labels_clean.csv")],
        class_label="subclass_label",
        sep=",",
        batch_size=256,
        index_col="cell",
        num_workers=32,
        deterministic=True,
        normalize=True,
        assume_numeric_label=False,
        currgenes=currgenes,
        refgenes=refgenes,
    )

    wandb_logger = WandbLogger(
        project=f"Human organoid benchmarking",
        name=f"human_organoid",
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    upload_callback = UploadCallback(path="checkpoints", desc="human_organoid_w_refgenes")

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
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
