import torch
from torch.utils.data import Dataset, DataLoader

import polars as pl
import positional as ps

from spatial_samplers import SpatialSampler, ExpoSpatialSampler


def senseiver_dataloader(data_config, num_workers=0, mode="train"):
    if mode == "validation":
        return DataLoader(
            ValidateLoader(data_config),
            batch_size=32,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers
        )
    elif mode == "train":
        return DataLoader(
            TrainLoader(data_config),
            batch_size=data_config["batch_size"],
            pin_memory=True,
            shuffle=True,
            num_workers=num_workers
        )
    elif mode == "test":
        return DataLoader(
            TestLoader(data_config),
            batch_size=128,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
        )
    elif mode == "inference":
        return DataLoader(
            InferenceLoader(data_config),
            batch_size=128,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
        )


class SenseiverLoader(Dataset):
    def __init__(self, data_config):
        data_path = data_config["data_path"]
        self.num_sensors = data_config["num_sensors"]
        self.num_queries = data_config["num_queries"]
        self.num_freqs = data_config["space_bands"]
        self.nlags = data_config["nlags"]
        self.locality = data_config["locality"]
        self.position_columns = data_config["position_columns"]  # to be embedded
        self.sensor_columns = data_config["sensor_columns"]
        self.query_columns = data_config["query_columns"]
        self.target_column = data_config["target_columns"]
        seed = data_config["seed"]

        dataset = pl.read_parquet(data_path)
        self.sensor_data, self.query_data, self.target = self.split_dataset(dataset)
        self.times = torch.tensor(list(set(self.query_data["time"])))

        self.convert_to_dicts()
        self.truncate_lags()  # Truncate lag columns

        self.sampler = ExpoSpatialSampler(
            self.sensor_data, self.query_data, data_config
        )

    def convert_to_dicts(self):
        """convert dataframe to dictionary of tensors"""
        sensor_cols = self.sensor_columns + self.position_columns
        query_cols = self.query_columns + self.position_columns
        self.sensor_data = {
            col: torch.Tensor(self.sensor_data[col]) for col in sensor_cols
        }
        self.query_data = {
            col: torch.Tensor(self.query_data[col]) for col in query_cols
        }

    def truncate_lags(self):
        """Truncate lagged data to nlags length"""
        lagged_columns = [
       "lags_conc",
       "lags_pm_prior",
       "lags_min_air_temperature",
       "lags_max_air_temperature", 
       "lags_min_relative_humidity",
       "lags_max_relative_humidity",
       "lags_wind_speed",
       "lags_precipitation_amount", 
       "lags_wind_direction"
   ]
        # Truncate each lagged column to nlags
        for col in lagged_columns:
            if col in self.sensor_data:
                self.sensor_data[col] = self.sensor_data[col][:, :self.nlags]

    def split_dataset(self, dataset):
        """

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        return None, None, None

    def __len__(self):
        return len(self.times)

    def get_sensor_data(self, idxs, query_idxs, time):
        """get sensor values, to be used in __getitem"""
        sensor_idxs = self.sampler.sample(idxs, query_idxs)
        sensor_positions = torch.cat(
            [
                self.sensor_data[pos][sensor_idxs].unsqueeze(1)
                for pos in self.position_columns
            ],
            dim=-1,
        )
        sensor_embs = ps.spatial_encoding(sensor_positions, self.num_freqs)
        sensor_values = torch.cat(
            [sensor_embs] +
            # [v if v.ndim > 0 else v.unsqueeze(0) for v in data.values()]
            [
                torch.cat(
                    [
                        self.sensor_data[feature][sensor_idxs]
                        if self.sensor_data[feature][sensor_idxs].ndim > 1
                        else self.sensor_data[feature][sensor_idxs].unsqueeze(1)
                        for feature in self.sensor_columns
                    ],
                    dim=-1,
                )
            ]
            + [torch.Tensor([time]).repeat(self.num_sensors).unsqueeze(1)],
            dim=-1,
        )
        return sensor_values

    def get_query_data(self, query_idxs):
        """get the query values, to be used in __getitem__"""
        query_positions = torch.cat(
            [
                self.query_data[pos][query_idxs].unsqueeze(1)
                for pos in self.position_columns
            ],
            dim=-1,
        )
        query_embs = ps.spatial_encoding(query_positions, self.num_freqs)
        query_values = torch.cat(
            [query_embs]
            + [
                torch.cat(
                    [
                        self.query_data[feature][query_idxs].unsqueeze(1)
                        for feature in self.query_columns
                    ],
                    dim=-1,
                )
            ],
            dim=-1,
        )
        return query_values

    def __getitem__(self, idx):
        time = 0
        idxs = 0
        query_idxs = 0
        sensor_values = self.get_sensor_data(idxs, query_idxs, time)
        query_values = self.get_query_data(query_idxs)
        field_values = self.target[query_idxs]
        return sensor_values, query_values, field_values.unsqueeze(-1)


class TrainLoader(SenseiverLoader):
    def __init__(self, data_config):
        """_summary_

        Args:
            data_config (_type_): _description_
        """
        super().__init__(data_config)

    def split_dataset(self, dataset):
        """

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        sensor_data = dataset.filter(pl.col("split") == "train")
        query_data = dataset.filter(pl.col("split") == "train")
        target = torch.Tensor(query_data[self.target_column])
        return sensor_data, query_data, target 

    def __getitem__(self, idx):
        time = self.times[idx]
        idxs = torch.where(self.sensor_data["time"] == time)[0]
        idxs = idxs[torch.randperm(idxs.shape[0])]
        query_idxs = idxs[0:1]
        idxs = idxs[1:]

        sensor_values = self.get_sensor_data(idxs, query_idxs, time)
        query_values = self.get_query_data(query_idxs)
        field_values = self.target[query_idxs]
        return sensor_values, query_values, field_values.unsqueeze(-1)


class ValidateLoader(SenseiverLoader):
    def __init__(self, data_config):
        """_summary_

        Args:
            data_config (_type_): _description_
        """
        super().__init__(data_config)

    def split_dataset(self, dataset):
        """

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        sensor_data = dataset.filter(pl.col("split") == "train")
        query_data = dataset.filter(pl.col("split") == "val")
        target = torch.Tensor(query_data[self.target_column])
        return sensor_data, query_data, target 

    def __getitem__(self, idx):
        time = self.query_data["time"][idx]
        idxs = torch.where(self.sensor_data["time"] == time)[0]
        query_idxs = torch.LongTensor([idx])
        sensor_values = self.get_sensor_data(idxs, query_idxs, time)
        query_values = self.get_query_data(query_idxs)
        field_values = self.target[query_idxs]
        return sensor_values, query_values, field_values.unsqueeze(-1)


class TestLoader(SenseiverLoader):
    def __init__(self, data_config):
        """_summary_

        Args:
            data_config (_type_): _description_
        """
        super().__init__(data_config)

    def split_dataset(self, dataset):
        """

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        sensor_data = dataset.filter(
            (pl.col("split") == "train") | (pl.col("split") == "val")
        )
        query_data = dataset.filter(pl.col("split") == "test")
        target = torch.Tensor(query_data[self.target_column])
        return sensor_data, query_data, target 

    def __getitem__(self, idx):
        time = self.query_data["time"][idx]
        idxs = torch.where(self.sensor_data["time"] == time)[0]
        query_idxs = torch.LongTensor([idx])
        sensor_values = self.get_sensor_data(idxs, query_idxs, time)
        query_values = self.get_query_data(query_idxs)
        field_values = self.target[query_idxs]
        return sensor_values, query_values, field_values.unsqueeze(-1)

    def __len__(self):
        return len(self.query_data["time"])

class InferenceLoader(SenseiverLoader):
    def __init__(self, data_config):
        """_summary_

        Args:
            data_config (_type_): _description_
        """
        super().__init__(data_config)

    def split_dataset(self, dataset):
        """

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        sensor_data = dataset.filter(
            (pl.col("split") == "train") | (pl.col("split") == "val")
        )
        query_data = dataset.filter(pl.col("split") == "test")
        return sensor_data, query_data, None

    def __len__(self):
        return len(self.query_data["lat"])

    def __getitem__(self, idx):
        time = self.query_data["time"][idx]
        idxs = torch.where(self.sensor_data["time"] == time)[0]
        query_idxs = torch.LongTensor([idx])
        sensor_values = self.get_sensor_data(idxs, query_idxs, time)
        query_values = self.get_query_data(query_idxs)
        return sensor_values, query_values, []
