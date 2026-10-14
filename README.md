# Lightweight Air Pollution Modeling using Deep Neural Network based Sparse Reconstruction

This project implements a spatial modeling approach for PM2.5 particulate matter using the Senseiver neural network architecture. The Senseiver reconstructs n-dimensional continuous signals from sparse observations, making it well-suited for environmental monitoring applications.


## Installation

```bash
# Clone the repository
git clone  <...>

# Install dependencies
pip install -r requirements.txt

```

## Quickstart
First, download the data:
```bash
python get_publication_data.py 
```

Train a model with the default configuration:

```bash
python train.py --config train.yaml 
```
Perform inference

```bash
python infer.py --config infer.yaml 
```

Sample configuration files are given in `sample_configs`.
## Data Availability

The model uses processed PM2.5 data paired with GRIDMET meteorological data. Raw and processed datasets are organized in the following directories:

- `data/raw`: Original, immutable data sources
- `data/processed`: Final, canonical data sets ready for modeling

## Configuration Options

The code is organized by separate configuration files for training a model and performing inference. 

### Training configuation
This is the most extensive configuration file. An example is given in `sample_configs/train_config.yaml`. 
The configuration system is set to be flexible in the passing of command line arguments.
So, we've implemented the following hiearchy for all parameters: (1) arguments specified at the command line (2) arguments specified in the config file (3) default arguments, which can be found in `src/models/s_parser.py`.

For example: 
```bash
python train.py --config train.yaml --lr .001
```
would overide the learning rate, `lr` specified in the `train.yaml` with the value `0.001`. 

There are many configuration parameters. Documented here are some of the most relevant, though feel free to reach about anything that is undocumented. 

- `--data_path`: Path to the processed PM2.5 data CSV file
- `--learning_rate`: Learning rate
- `--batch_size`: Batch size
- `--num_sensors`: Number of sensor inputs (default: 8)
- `--num_queries`: Number of query points to predict (default: 2)
- `--seed`: Random seed for reproducibility (default: 123)
- `--enc_preproc`: Encoder preprocessing channels (default: 16)
- `--dec_num_latent_channels`: Decoder latent channels (default: 16)
- `--enc_num_latent_channels`: Encoder latent channels (default: 16)
- `--dec_preproc_ch`: Decoder preprocessing channels (default: 16)
- `--dec_num_latent_ch`: Decoder preprocessing channels (default: 16)
- `--nlags`: Number of time lags to include in any lagged variables (default: 1)
- `--sample_closest`: Only use the closest sensors to the query for inference.
- `--target_columns` : Name of the column that we want to predict
- `--sensor_columns`: Options are listed below. 🚩 🚩 🚩 Currently, the implementation requires `land_cover` and `time` to be the last features listed, as these features are encoded prior to being passed to the architecture 🚩 🚩 🚩.
  - `conc`
  - `min_air_temperature`
  - `max_air_temperature`
  - `min_relative_humidity`
  - `max_relative_humidity`
  - `wind_speed`
  - `precipitation_amount`
  - `wind_direction`
  - `lags_conc`
  - `lags_min_air_temperature`
  - `lags_max_air_temperature`
  - `lags_min_relative_humidity`
  - `lags_max_relative_humidity`
  - `lags_wind_speed`
  - `lags_precipitation_amount`
  - `lags_wind_direction`
  - `day_population`
  - `night_population`
  - `land_cover`
  - `time`
- `--query_columns`: Data that are available at the query location. 🚩 🚩 🚩 Currently, the implementation requires `land_cover` and `time` to be the last features listed, as these features are encoded prior to being passed to the architecture 🚩 🚩 🚩.
  - `min_air_temperature`
  - `max_air_temperature`
  - `min_relative_humidity`
  - `max_relative_humidity`
  - `wind_speed`
  - `precipitation_amount`
  - `wind_direction`
  - `day_population`
  - `night_population`
  - `land_cover`
  - `time`
- `--position_columns:` _Special_ columns that go through spatial encoding. Available and sensors and queries.
  - `lat`
  - `lon `
  - `elevation`

### Inference
Similar to the training configuration file, inference is both command-line and config friendly, eg: 
```bash
python infer.py --config infer.yaml --
```
Parameters include; 
- `--model_path`: Path to pretrained model checkpoint
- `--train_config_path`: Path to the config file associate with the pretrained model.  
- `--data_path`:  Path to the inference dataset. This should have a column `split`. The rows with the `split==train` and `split==val` are taken to be the sensor data, and therefore must have `target_col` values. The rows with `split==test` are taken to be the inference points, and do not require `target_col` to be populated. 
- `--save_path`: Where to save the output parquet path.  

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
