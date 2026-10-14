import os
import glob
import yaml
import wandb
import torch
import argparse
import pandas as pd
from train import train
from valid_utils import run_evaluation, run_inference
import infer
from lightning.pytorch.loggers import WandbLogger
from yearly_test import calculate_yearly_r2
from map_predictions import plot_predictions_map
from multiscale_analysis import generate_r2_mae_maps

torch.set_float32_matmul_precision('high')

# Define output folder for maps and tables
OUTPUT_MAPS_DIR = "../../models/output_maps"
os.makedirs(OUTPUT_MAPS_DIR, exist_ok=True)


def load_save_path(config_path):
    """Load the save_path from a configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("save_path", "")


def get_config():
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="full pipeline parser")

    # Add config file as an argument
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to the pipeline configuration file. Default is pipeline_config.yaml",
    )

    args, unknown = parser.parse_known_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    return config


def generate_inference_maps(inference_save_path, logger):
    plot_predictions_map(inference_save_path)
    predictions_artifact = wandb.Artifact("predictions_map", type="map_visualization")
    for ext in ["png", "eps", "pdf"]:
        map_file = os.path.join(OUTPUT_MAPS_DIR, f"predictions_map.{ext}")
        if os.path.exists(map_file):
            predictions_artifact.add_file(map_file)
    logger.experiment.log_artifact(predictions_artifact)


def generate_test_artifacts(test_save_path, logger):
    r2_mae_maps = generate_r2_mae_maps(test_save_path, output_dir=OUTPUT_MAPS_DIR)
    for metric, img_path in r2_mae_maps.items():
        if img_path.endswith(".png") and os.path.exists(img_path):
            logger.experiment.log({f"Map of {metric}": wandb.Image(img_path)})
            metric_artifact = wandb.Artifact(f"{metric}_map", type="map_visualization")
            for ext in ["png", "eps", "pdf"]:
                map_file = os.path.join(OUTPUT_MAPS_DIR, f"{metric}_map.{ext}")
                if os.path.exists(map_file):
                    metric_artifact.add_file(map_file)
            logger.experiment.log_artifact(metric_artifact)

    # Calculate and log yearly R² scores
    yearly_r2_csv, _ = calculate_yearly_r2(test_save_path, output_dir=OUTPUT_MAPS_DIR)
    yearly_r2_df = pd.read_csv(yearly_r2_csv)
    logger.experiment.log({"Yearly R² Scores": wandb.Table(dataframe=yearly_r2_df)})


def main():
    logger = WandbLogger(project="pollution")
    run_config = get_config()
    
    # Train model
    print("Starting training...")
    _, model_path = train(run_config["train_config"], logger=logger)

    # Evaluate on the test set
    run_evaluation(
        model_path,
        run_config["train_config"],
        run_config["test_data_path"],
        run_config["test_save_path"],
    )
    generate_test_artifacts(run_config["test_save_path"], logger)

    # Run inference
    run_inference(
        model_path,
        run_config["train_config"],
        run_config["inference_data_path"],
        run_config["inference_save_path"],
    )
    generate_inference_maps(run_config["inference_save_path"], logger)


if __name__ == "__main__":
    main()
