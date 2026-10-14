import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from glob import glob as gb
import wandb
import torch
import torch.multiprocessing as mp

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from s_parser import parse_args
from dataloaders import senseiver_dataloader
from network_light import Senseiver

torch.set_float32_matmul_precision('high')

def train(config=None, logger=None):

    data_config, encoder_config, decoder_config = parse_args(config)
    # setup/disable wandb logging
    if data_config['debug']:
        os.environ['WANDB_MODE'] = 'dryrun'
    else:
        os.environ['WANDB_MODE'] = 'online'


    # Initialize WandbLogger
    if logger is None: 
        logger = WandbLogger(project="pollution")
    run_id = logger.experiment.id  
    chkpt_path = data_config['checkpoint_path']

    # load the simulation data and create a dataloader
    # num_workers = mp.cpu_count() // 2
    # print(f'Using {num_workers} workers'.format(num_workers))
    num_workers = 0
    train_dataloader = senseiver_dataloader(data_config, num_workers=num_workers, mode='train')
    val_dataloader = senseiver_dataloader(data_config, num_workers=num_workers, mode='validation')

    # instantiate new Senseiver
    model = Senseiver(
        **encoder_config,
        **decoder_config,
        **data_config
    )
    
    # Setup ModelCheckpoint callback to monitor validation metric
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        filename="val-{epoch:02d}",
        save_top_k=1,
        mode="min", 
        dirpath=os.path.join(chkpt_path,run_id)
    )

    # callbacks
    cbs = [checkpoint_callback,
            EarlyStopping(monitor="val_mae", check_finite=False, patience=100)]


    trainer = Trainer(max_epochs=-1,
                        callbacks=cbs,
                        accumulate_grad_batches=data_config['accum_grads'],
                        # log_every_n_steps=data_config['num_batches'],
                        log_every_n_steps=10,
                        limit_train_batches=1000,
                        limit_val_batches=200,
                        logger=logger,
                        precision=32, 
                        accelerator='gpu',
                        devices=1
                        )

    
    trainer.fit(model, train_dataloader, val_dataloader
                )

    return trainer.logged_metrics['val_mae'].item(), checkpoint_callback.best_model_path

if __name__=='__main__':
    # arg parser
    _ = train()
