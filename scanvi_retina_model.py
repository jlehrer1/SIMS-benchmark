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
from sklearn.metrics import accuracy_score, f1_score 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from networking import download 

args = parser.parse_args()
lr, weight_decay, name, test = args.lr, args.weight_decay, args.name, args.test 

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
data.X = data.X.todense()
labels = pd.read_csv(join(here, 'retina_labels_numeric.csv'), index_col="cell")

