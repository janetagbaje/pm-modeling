import argparse
import yaml 
from valid_utils import run_inference

def load_inference_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def infer(config=None):
    if config is not None:
        config = load_inference_config(config)
    else:
        parser = argparse.ArgumentParser(description="Run inference on a model")
    
        # Add config file as an argument
        parser.add_argument('--config', type=str, default='../../sample_configs/inference_config.yaml',
                            help='Path to the inference configuration file. Default is inference_config.yaml')
        
        # Parse known arguments so we can load the config file first
        args, unknown = parser.parse_known_args()

        config = load_inference_config(args.config)

    # Now add the other arguments, using values from the config file as defaults
    parser.add_argument('-m', '--model_path', type=str, default=config.get('model_path'),
                        help='Path to the model file.')
    parser.add_argument('-d', '--data_path', type=str, default=config.get('data_path'),
                        help='Path to the dataset.')
    parser.add_argument('-o', '--save_path', type=str, default=config.get('save_path'),
                        help='Path to save the results.')
    parser.add_argument('-t', '--train_config_path', type=str, default=config.get('train_config_path'),
                        help='Path to config file for model.')
    
    # Reparse with the full set of arguments (including possible command line overrides)
    args = parser.parse_args()
    config['model_path'] = args.model_path
    config['data_path'] = args.data_path
    config['save_path'] = args.save_path
    config['train_config_path'] = args.train_config_path

    print(config)

    # Call the run_inference function with parsed arguments
    run_inference(config['model_path'], config['train_config_path'], config['data_path'], config['save_path'])

if __name__ == "__main__":
   infer()