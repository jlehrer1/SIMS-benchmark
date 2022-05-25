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
import sys
import anndata as an
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from networking import download 

here = pathlib.Path(__file__).parent.resolve()

# Download training data
for file in ['mouse_labels.csv', 'mouse_clipped.h5ad']:
    print(f'Downloading {file}')

    if not os.path.isfile(file):
        download(
            remote_name=join('jlehrer', 'mouse_benchmark', file),
            file_name=join(here, file),
        )

# Set up the data for scVI
data = an.read_h5ad(join(here, 'mouse_clipped.h5ad'))
data.X = data.X.todense()

labels = pd.read_csv(join(here, 'mouse_labels.csv'), index_col='cell')

mouse_data = data[labels.index.values, :]
mouse_data.obs["label"] = mouse_data.obs["subclass_label"].values
mouse_data.obs["label"] = pd.Series(labels["subclass_label"], dtype="category")
mouse_data.obs = mouse_data.obs.reset_index()

# Set up train/val/test split same as SIMS model 
indices = mouse_data.obs.loc[:, 'subclass_label']
train, val = train_test_split(indices, test_size=0.2, random_state=42, stratify=indices)
train, test = train_test_split(train, test_size=0.2, random_state=42, stratify=train)

train_data = mouse_data[train.index, :]
valid_data = mouse_data[val.index, :]
test_data = mouse_data[test.index, :]

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
    labels_key="subclass_label",
    unlabeled_category="N/A", # All are labeled, so we ignore this 
)

lvae.train(
    max_epochs=100, 
)

# Now get the test accuracy
# Also use a PyTorch logger to we can visualize the results 
logger = WandbLogger(
    project='scANVI Comparison',
    name='Mouse Model (Allen Brain Institute Data)'
)
preds = lvae.predict(test_data)
truth = test_data.obs['subclass_label'].values

acc = accuracy_score(preds, truth)
logger.log("accuracy", acc)

f1 = f1_score(preds, truth, average=None)
mf1 = np.nanmedian(f1)

logger.log("Median f1", mf1)