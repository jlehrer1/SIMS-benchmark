import os
import pathlib
import sys
import anndata as an
import torch
import argparse

from os.path import join, dirname, abspath

sys.path.append(join(dirname(abspath(__file__)), ".."))

import pandas as pd
import numpy as np
import anndata as an
import sys, os

sys.path.append("../src")

import sys
import os
import pathlib
from typing import *

import torch
import numpy as np
import pandas as pd
import anndata as an

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from scsims import *
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
    data_path = join(here, "..", "data", "cd4_atlas")

    print("Making data folder")
    os.makedirs(data_path, exist_ok=True)

    for file in [
        "GSE99254_108989_96838_filtered_QC.h5ad",
        "GSE_integrated_0609_count.h5ad",
        "Atlas_Annotation_CD4.csv",
    ]:
        print(f"Downloading {file}")

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join("jlehrer", "cd4_atlas", file),
                file_name=join(data_path, file),
            )

    train = an.read_h5ad(join(data_path, "GSE99254_108989_96838_filtered_QC.h5ad"), backed="r+")
    test = an.read_h5ad(join(data_path, "GSE_integrated_0609_count.h5ad"), backed="r+")

    # currgenes = train.var.index.values
    # refgenes = list(set(train.var.index.values).intersection(test.var.index.values))

    module = DataModule(
        datafiles=[join(data_path, "GSE99254_108989_96838_filtered_QC.h5ad")],
        labelfiles=[join(data_path, "Atlas_Annotation_CD4.csv")],
        class_label="Atlas Annotation",
        batch_size=256,
        num_workers=32,
        deterministic=True,
        # currgenes=currgenes,
        # refgenes=refgenes,
        # preprocess=True,
    )

    wandb_logger = WandbLogger(
        project=f"CD4 Atlas",
        name=f"RAW-{name}",
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    upload_callback = UploadCallback(path="checkpoints", desc=f"cd4_atlas_raw_counts_intersection_{name}")

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
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

    module.prepare_data()
    module.setup()

    model = SIMSClassifier(
        input_dim=module.num_features,
        output_dim=module.num_labels,
        weights=module.weights,
    )

    trainer.fit(model, datamodule=module)
    # trainer.test(model, datamodule=module)
