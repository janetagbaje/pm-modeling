import os
import argparse
import torch
import yaml


# function to parse boolean args
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def load_train_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args(config=None):
    parser = argparse.ArgumentParser(description="Senseiver trainer")
    parser.add_argument('--config', type=str, default='train_config.yaml',
                        help='Path to the train configuration file. Default is train_config.yaml')

    args, unknown = parser.parse_known_args()
    if config is not None:
        args.config = config
    config = load_train_config(args.config)

    # Data
    parser.add_argument("--data_path", default=config.get('data_path'), type=str)
    parser.add_argument("--checkpoint_path", default=config.get('checkpoint_path'), type=str)
    parser.add_argument("--split_path", default=config.get('split_path'), type=str)
    parser.add_argument("--split", default=config.get('split'), type=str)
    parser.add_argument("--debug", default=config.get('debug'), type=str2bool)
    parser.add_argument("--num_sensors", default=config.get('num_sensors'), type=int)
    parser.add_argument("--num_queries", default=config.get('num_queries'), type=int)
    parser.add_argument("--gpu_device", default=config.get('gpu_device'), type=int)
    parser.add_argument("--seed", default=config.get('seed'), type=int)
    parser.add_argument("--batch_size", default=config.get('batch_size'), type=int)
    parser.add_argument("--lr", default=config.get('lr'), type=float)
    parser.add_argument("--accum_grads", default=config.get('accum_grads'), type=int)
    parser.add_argument("--locality", default=config.get('locality'), type=float)
    parser.add_argument("--nlags", default=config.get('nlags'), type=int)
    parser.add_argument("--sample_closest", default=config.get('sample_closest'), type=bool)
    parser.add_argument("--sensor_columns", default=config.get('sensor_columns'), type=list)
    parser.add_argument("--query_columns", default=config.get('query_columns'), type=list)
    parser.add_argument("--position_columns", default=config.get('position_columns'), type=list)
    parser.add_argument("--target_columns", default=config.get('target_columns'), type=str)

    # Positional Encodings
    parser.add_argument("--space_bands", default=config.get('space_bands'), type=int)

    # Encoder
    parser.add_argument("--enc_preproc_ch", default=config.get('enc_preproc_ch'), type=int)
    parser.add_argument("--num_latents", default=config.get('num_latents'), type=int)
    parser.add_argument("--enc_num_latent_channels", default=config.get('enc_num_latent_channels'), type=int)
    parser.add_argument("--num_layers", default=config.get('num_layers'), type=int)
    parser.add_argument("--num_cross_attention_heads", default=config.get('num_cross_attention_heads'), type=int)
    parser.add_argument("--enc_num_self_attention_heads", default=config.get('enc_num_self_attention_heads'), type=int)
    parser.add_argument("--num_self_attention_layers_per_block", default=config.get('num_self_attention_layers_per_block'), type=int)
    parser.add_argument("--dropout", default=config.get('dropout'), type=float)

    # Decoder
    parser.add_argument("--dec_preproc_ch", default=config.get('dec_preproc_ch'), type=int)
    parser.add_argument("--dec_num_latent_channels", default=config.get('dec_num_latent_channels'), type=int)
    parser.add_argument("--dec_num_cross_attention_heads", default=config.get('dec_num_cross_attention_heads'), type=int)

    args = parser.parse_args()
    return assign_args(args)

def count_channels(channel_list, nlags):
    lag_count = 0
    for channel in channel_list:
        if 'lag' in channel:
            lag_count +=1 
    channels = len(channel_list)-lag_count + lag_count*nlags
    return channels

def assign_args(args):
    if torch.cuda.is_available():
        accelerator = "gpu"
        gpus = [args.gpu_device]
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        gpus = [args.gpu_device]
    else:
        accelerator = "cpu"
        gpus = None

    sensor_channels = count_channels(args.sensor_columns, args.nlags)
    query_channels = count_channels(args.query_columns, args.nlags)
    # Assign the args
    data_config = dict(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        split = args.split,
        debug = args.debug,
        num_sensors=args.num_sensors,
        num_queries=args.num_queries,
        gpu_device=None if accelerator == "cpu" else gpus,
        accelerator=accelerator,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        accum_grads=args.accum_grads,
        space_bands=args.space_bands,
        locality = args.locality,
        nlags = args.nlags,
        sensor_channels = sensor_channels,
        query_channels = query_channels,
        sample_closest = args.sample_closest, 
        sensor_columns = args.sensor_columns,
        query_columns = args.query_columns,
        position_columns = args.position_columns,
        target_columns = args.target_columns
    )

    encoder_config = dict(
        enc_preproc_ch=args.enc_preproc_ch,  # expand input dims
        num_latents=args.num_latents,  # "seq" latent
        enc_num_latent_channels=args.enc_num_latent_channels,  # channels [b,seq,chan]
        num_layers=args.num_layers,
        num_cross_attention_heads=args.num_cross_attention_heads,
        enc_num_self_attention_heads=args.enc_num_self_attention_heads,
        num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
        dropout=args.dropout,
    )

    decoder_config = dict(
        dec_preproc_ch=args.dec_preproc_ch,  # latent bottleneck
        dec_num_latent_channels=args.dec_num_latent_channels,  # hyperparam
        latent_size=1,  # collapse from n_sensors to 1 observation
        dec_num_cross_attention_heads=args.dec_num_cross_attention_heads,
    )

    return data_config, encoder_config, decoder_config

