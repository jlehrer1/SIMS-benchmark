import scvi
import scvi.data
import boto3
from typing import *
import pandas as pd 
import numpy as np
import anndata as an

import scvi
import scvi.data
import pandas as pd 
import numpy as np

import os
from os.path import join 
import pathlib 
import anndata as an
from sklearn.model_selection import train_test_split 
from sklearn.metrics import (
    accuracy_score, 
    f1_score,
    precision_score,
    recall_score,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from networking import download 

here = pathlib.Path(__file__).parent.resolve()

for file in ['retina_T.h5ad', 'retina_labels_numeric.csv']:
    print(f'Downloading {file}')

    if not os.path.isfile(join(here, file)):
        download(
            remote_name=join('jlehrer', 'retina', file),
            file_name=join(here, file),
        )

# Set up the data for scVI
data = an.read_h5ad(join(here, 'retina_T.h5ad'))
labels = pd.read_csv(join(here, 'retina_labels_numeric.csv'), index_col="cell")

data.obs = data.obs.reset_index()
data.obs["categorical_class_label"] = pd.Series(labels["categorical_class_label"].astype(str), dtype="category")

# Set up train/val/test split same as SIMS model 
indices = data.obs.loc[:, 'categorical_class_label']
train, val = train_test_split(indices, test_size=0.2, random_state=42, stratify=indices)
train, test = train_test_split(train, test_size=0.2, random_state=42, stratify=train)

train_data = data[train.index, :]
valid_data = data[val.index, :]
test_data = data[test.index, :]

# Train the scVI model
train_data = train_data.copy()
scvi.model.SCVI.setup_anndata(train_data)
vae = scvi.model.SCVI(train_data, n_layers=2, n_latent=30, gene_likelihood="nb")

vae.train(
    early_stopping=True,
    max_epochs=150,
    early_stopping_patience=5,
)

# Train the scANVI model
lvae = scvi.model.SCANVI.from_scvi_model(
    vae,
    adata=train_data,
    labels_key="categorical_class_label",
    unlabeled_category="N/A", # All are labeled, so we ignore this 
)

lvae.train(
    max_epochs=100, 
)

# Now get the test accuracy
# Also use a PyTorch logger to we can visualize the results 
logger = WandbLogger(
    project='scANVI Comparison',
    name='Retina Model (Allen Brain Institute Data)'
)
preds = lvae.predict(test_data)
truth = test_data.obs['categorical_class_label'].values

acc = accuracy_score(preds, truth)
f1 = f1_score(preds, truth, average=None)
mf1 = np.nanmedian(f1)

precision = precision_score(preds, truth, average="macro")
recall = recall_score(preds, truth, average="macro")

logger.log_metrics({
    "Accuracy": acc,
    "Median F1": mf1,
    "Precision": precision,
    "Recall": recall,
})


