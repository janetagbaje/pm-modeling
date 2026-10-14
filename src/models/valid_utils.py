import os
import sys
from argparse import Namespace
import yaml
from pathlib import Path
import pandas as pd
import torch

from tqdm import tqdm

import network_light as nl
import s_parser as sp
import dataloaders as dl


def get_train_config(yaml_file_path):
    """_summary_

    Args:
        config_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(yaml_file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    for key, val in config.items():
        try:
            config[key] = config[key]["value"]
        except:
            continue
    args = Namespace(**config)
    data_config, encoder_config, decoder_config = sp.assign_args(args)
    return args, data_config, encoder_config, decoder_config


def load_model(model_id, epoch):
    """_summary_

    Args:
        model_id (_type_): _description_
        epoch (_type_): _description_

    Returns:
        _type_: _description_
    """

    CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
    path_to_checkpoint = (
    CHECKPOINT_DIR / model_id / "checkpoints" / f"val-epoch={epoch}.ckpt"
    )

    model = nl.Senseiver.load_from_checkpoint(path_to_checkpoint)
    return model

def get_model_results(model, dataloader):
    """_summary_

    Args:
        model (_type_): _description_
        dataloader (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_pred = []
    sensor_means = []

    model.eval()
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():
        for batch in tqdm(dataloader):
            sensors, query_coords, _ = batch

            # Forward pass through the model
            out = model(sensors.to(device), query_coords.to(device))
            
            # Ensure the output shape is compatible
            out = out.view(out.size(0), -1)  # Reshape to (batch_size, num_outputs)
            
            # Populate all_pred and sensor_means
            all_pred += list(out.detach().cpu().numpy().ravel())
            sensor_means += list(sensors[:, 0].mean(dim=1).detach().cpu().numpy().ravel())  # Calculate means for each sample in the batch

    print(f"Total predictions: {len(all_pred)}")
    print(f"Total sensor means: {len(sensor_means)}")
    return all_pred, sensor_means

def run_evaluation(model_path, config_path, data_path, save_path):
    run_inference(model_path, config_path, data_path, save_path, mode='test')

def run_inference(model_path, config_path, data_path, save_path, mode='inference'):
    """_summary_

    Args:
        model_path (_type_): _description_
        config_path (_type_): _description_
        data_path (_type_): _description_
        split_path (_type_): _description_
        save_path (_type_): _description_
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    # model = nl.Senseiver.load_from_checkpoint(model_path, map_location=device)

    # _, data_config, _, _ = get_train_config(config_path)
    _,data_config, encoder_config, decoder_config = get_train_config(config_path)
    data_config['data_path'] = data_path 

    model = nl.Senseiver(**encoder_config, **decoder_config, **data_config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)

    loader = dl.senseiver_dataloader(data_config, mode=mode)
    model_pred, sensor_means = get_model_results(model, loader)
    write_results(data_config, model_pred, sensor_means, save_path)


def write_results(data_config, model_pred, sensor_means, save_path):
    """_summary_

    Args:
        data_config (_type_): _description_
        model_pred (_type_): _description_
        sensor_means (_type_): _description_
        save_path (_type_): _description_
    """
    query_data = pd.read_parquet(data_config['data_path'])
    print(query_data)
    query_data = query_data[query_data['split']=='test']
    query_data['pred'] = model_pred
    query_data['sensor_means'] = sensor_means
    query_data.to_csv(save_path)
    print(f"Saved results to {save_path}")

