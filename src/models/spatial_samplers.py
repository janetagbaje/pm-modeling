import torch
import numpy as np


class SpatialSampler:
    def __init__(self, sensor_data, query_data, config):
        self.sensor_data = sensor_data
        self.query_data = query_data
        self.locality = config['locality']
        self.num_sensors = config['num_sensors']

    def apply_locality(self, idxs, query_idx):
        lat_inds = (self.sensor_data["lat"][idxs] < self.query_data["lat"][query_idx]+self.locality) & (self.sensor_data["lat"][idxs] > self.query_data["lat"][query_idx]-self.locality)
        lon_inds = (self.sensor_data["lon"][idxs] < self.query_data["lon"][query_idx]+self.locality) & (self.sensor_data["lon"][idxs] > self.query_data["lon"][query_idx]-self.locality)
        local_inds = lat_inds*lon_inds
        return local_inds

    def sample(self, idxs, query_idx):
        local_inds = self.apply_locality(idxs, query_idx)
        idxs_local = idxs[local_inds]
        if len(idxs_local) == 0:  # just random
            return idxs[torch.randperm(idxs.shape[0])][:self.num_sensors]

        if len(idxs_local) < self.num_sensors:
            idxs_local = torch.cat(self.num_sensors*[idxs_local])

        idxs_sensors = idxs_local[torch.randperm(idxs_local.shape[0])][:self.num_sensors]
        return idxs_sensors
    
class ExpoSpatialSampler:
    def __init__(self, sensor_data, query_data, config):
        self.sensor_data = sensor_data
        self.query_data = query_data
        self.locality = config['locality']
        self.num_sensors = config['num_sensors']
        self.closest = config['sample_closest']
        self.max_scale = .21 # hardcoded

    def apply_locality(self, idxs, query_idx, max_scale):
        lat_inds = (self.sensor_data["lat"][idxs] < self.query_data["lat"][query_idx]+max_scale) & (self.sensor_data["lat"][idxs] > self.query_data["lat"][query_idx]-max_scale)
        lon_inds = (self.sensor_data["lon"][idxs] < self.query_data["lon"][query_idx]+max_scale) & (self.sensor_data["lon"][idxs] > self.query_data["lon"][query_idx]-max_scale)
        local_inds = lat_inds*lon_inds
        return local_inds

    def get_weights(self, local_inds, query_idx, variance=1):
        lats = self.sensor_data["lat"][local_inds]
        lons = self.sensor_data["lon"][local_inds]
        distances_2 = torch.sqrt( (lats-self.query_data["lat"][query_idx])**2 
                                + (lons-self.query_data["lon"][query_idx])**2 )
        return torch.exp( -distances_2 / variance )

    def sample(self, idxs, query_idx):
        local_inds = self.apply_locality(idxs, query_idx, self.max_scale)
        idxs_local = idxs[local_inds]
        if len(idxs_local) < self.num_sensors:
            local_inds = self.apply_locality(idxs, query_idx, 1)
            idxs_local = idxs[local_inds]

        weights = self.get_weights(idxs_local, query_idx, self.locality)
        if self.closest:
            sorted_idx_local = idxs_local[weights.argsort(descending=True)]
            return sorted_idx_local[:self.num_sensors]
        else:
            weights = np.array(weights)
            return torch.LongTensor(np.random.choice(idxs_local, self.num_sensors, 
                                                    p=weights/weights.sum(), replace=True))

