import os
import pathlib 
import torch 
import argparse 
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger

from scsims import *
from os.path import join 
from networking import * 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        '--test',
        action='store_true',
        required=False
    )

    args = parser.parse_args()
    name, test = args.name, args.test 

    here = pathlib.Path(__file__).parent.resolve()
    data_path = join(here, '..', 'data', 'retina')

    print('Making data folder')
    os.makedirs(data_path, exist_ok=True)

    for file in ['retina_T.h5ad', 'retina_labels_numeric.csv']:
        print(f'Downloading {file}')

        if not os.path.isfile(join(data_path, file)):
            download(
                remote_name=join('jlehrer', 'retina', file),
                file_name=join(data_path, file),
            )

    # Define labelfiles and trainer
    datafiles=[join(data_path, 'retina_T.h5ad')]
    labelfiles=[join(data_path, 'retina_labels_numeric.csv')]

    device = ('cuda:0' if torch.cuda.is_available() else None)

    module = DataModule(
        datafiles=datafiles,
        labelfiles=labelfiles,
        class_label='class_label',
        index_col='cell',
        batch_size=16,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        normalize=True,
        deterministic=True,
    )

    wandb_logger = WandbLogger(
        project=f"Retina Model",
        name=name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    upload_callback = UploadCallback(
        path='checkpoints',
        desc='retina'
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
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
        ]
    )

    if not test:
        model = SIMSClassifier(
            input_dim=module.num_features,
            output_dim=module.num_labels,
        )

        # train model
        trainer.fit(model, datamodule=module)
        trainer.test(model, datamodule=module)
    else:  
        checkpoint_path = join(here, '..', 'checkpoints/checkpoint-80-desc-retina.ckpt')

        if not os.path.isfile(checkpoint_path):
            os.makedirs(join(here, '..', 'checkpoints'), exist_ok=True)
            download(
                'jlehrer/model_checkpoints/checkpoint-80-desc-retina.ckpt',
                checkpoint_path
            )

        model = SIMSClassifier.load_from_checkpoint(
            checkpoint_path,
            input_dim=module.input_dim,
            output_dim=module.output_dim,
            n_d=32,
            n_a=32,
            n_steps=10,
        )

        trainer.test(model, datamodule=module)
