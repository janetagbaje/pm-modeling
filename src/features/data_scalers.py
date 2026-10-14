import numpy as np
import json

import numpy as np
import json

class Scaler:
    """
    Class for all scalers.

    Methods:
        fit(X): Compute necessary statistics for scaling.
        transform(X): Transform the data.
        fit_transform(X): Fit and transform.
        inverse_transform(X): Reverse the transformation.
        save_scaler(file_path): Save scaler parameters.
        load_scaler(file_path): Load scaler parameters.
    """
    def __init__(self):
        pass
    
    def fit(self, X):
        raise NotImplementedError("Subclasses must implement the 'fit' method.")
    
    def transform(self, X, y=None):
        raise NotImplementedError("Subclasses must implement the 'transform' method.")
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        raise NotImplementedError("Subclasses must implement the 'inverse_transform' method.")
    
    def save_scaler(self, file_path):
        """Save the scaler parameters to a file."""
        scaler_params = self._get_params()
        with open(file_path, 'w') as f:
            json.dump(scaler_params, f)
    
    def load_scaler(self, file_path):
        """Load the scaler parameters from a file."""
        with open(file_path, 'r') as f:
            scaler_params = json.load(f)
        self._set_params(scaler_params)
    
    def _get_params(self):
        """This method will be defined by subclasses to get specific parameters."""
        raise NotImplementedError("Subclasses must implement '_get_params' method.")
    
    def _set_params(self, params):
        """This method will be defined by subclasses to set specific parameters."""
        raise NotImplementedError("Subclasses must implement '_set_params' method.")

class MinMaxScaler(Scaler):
    """
    Scales features to a given range [min, max].

    Args:
        column (str): Column name to scale.
        feature_range (tuple): Range of transformed data (default=(0,1)).

    Methods:
        fit(X): Compute min and max for scaling.
        transform(X): Apply min-max scaling.
        inverse_transform(X): Reverse the scaling.
    """
    def __init__(self, column, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range
        self.column_ = column
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """Compute the min and max to be used for later scaling."""
        X = X[self.column_].to_numpy()
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.max_ - self.min_)

    def transform(self, X):
        """Scale the data according to the computed min and max."""
        X_t = X.copy()
        X_t[self.column_] = (X_t[self.column_] - self.min_) * self.scale_ + self.feature_range[0]
        return X_t

    def inverse_transform(self, X_scaled):
        """Undo the scaling."""
        X_t = X_scaled.copy()
        X_t[self.column_] = (X_t[self.column_] - self.feature_range[0]) / self.scale_ + self.min_
        return X_t

    def _get_params(self):
        return {
            'min': self.min_.tolist(),
            'max': self.max_.tolist(),
            'scale': self.scale_.tolist(),
            'column': self.column_.tolist(),
            'feature_range': self.feature_range
        }

    def _set_params(self, params):
        self.min_ = np.array(params['min'])
        self.max_ = np.array(params['max'])
        self.scale_ = np.array(params['scale'])
        self.column_ = np.array(params['column'])
        self.feature_range = tuple(params['feature_range'])

class StandardScaler(Scaler):
    """
    Standardizes features by computing the mean and scaling to unit variance.

    Args:
        column (str): Column name to standardize.

    Methods:
        fit(X): Compute mean and standard deviation.
        transform(X): Apply standardization.
        inverse_transform(X): Reverse standardization.
    """
    def __init__(self,column):
        super().__init__()
        self.column_ = column
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """Compute the mean and standard deviation to be used for later scaling."""
        X = X[self.column_].to_numpy()
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        """Standardize the data (z-score normalization)."""
        X_t = X.copy()
        X_t[self.column_]=(X_t[self.column_] - self.mean_) / self.std_
        return X_t

    def inverse_transform(self, X_scaled):
        """Undo the standardization."""
        X_t = X_scaled.copy()
        X_t[self.column_] = X_t[self.column_] * self.std_ + self.mean_
        return X_t

    def _get_params(self):
        return {
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist()
        }

    def _set_params(self, params):
        self.mean_ = np.array(params['mean'])
        self.std_ = np.array(params['std'])

class LatLonScaler(Scaler):
    """
    Normalizes latitude and longitude into 3D unit vectors (Cartesian coordinates).

    Args:
        lat_column (str): latitude column.
        lon_column (str): longitude column.

    Methods:
        transform(X): Convert lat/lon to normalized Cartesian coordinates.
        inverse_transform(X): if needed.
    """
    def __init__(self, lat_column, lon_column):
        super().__init__()
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.params = {}  # You can store fit parameters here if needed
    
    def fit(self, X, y=None):
        """
        No special fitting required for normalization in this context, but this
        method could be used for calculating bounds or other transformations.
        
        Parameters:
        X (array-like): List or array of latitude and longitude pairs.
        """
        # In this case, no specific fitting process is required.
        pass
    
    def transform(self, X):
        """
        Transforms a list of latitude and longitude pairs to normalized
        Cartesian coordinates.

        Parameters:
        X (array-like): List or array of latitude and longitude pairs.

        Returns:
        np.array: Normalized Cartesian coordinates.
        """
        transformed_coords = []
        X_t = X.copy()
        X = X_t[[self.lat_column,self.lon_column]].to_numpy()
        xs, ys = [],[]
        for lat, lon in X:
            x, y, z = self.lat_lon_to_cartesian(lat, lon)
            norm = np.sqrt(x**2 + y**2 + z**2)
            xs.append(x/norm)
            ys.append(y/norm)
            # transformed_coords.append([x/norm, y/norm, z/norm])

        X_t[self.lat_column] = xs
        X_t[self.lon_column] = ys
        return X_t

    def inverse_transform(self, X):
        """
        Since we are normalizing lat/lon into 3D Cartesian coordinates,
        this would typically be difficult to reverse directly without extra information.
        For now, this method will not be implemented.
        """
        raise NotImplementedError("Inverse transform not implemented.")

    def lat_lon_to_cartesian(self, lat, lon):
        """
        Converts latitude and longitude to 3D Cartesian coordinates on a unit sphere.

        Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

        Returns:
        tuple: Cartesian coordinates (x, y, z).
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        return x, y, z

    def _get_params(self):
        """Returns parameters for saving. In this case, nothing to store."""
        return self.params
    
    def _set_params(self, params):
        """Loads parameters from file."""
        self.params = params

class LogarithmicScaler(Scaler):
    """
    Applies logarithmic transformation to the specified feature.

    Args:
        column (str): feature name to scale.
        base (float): Base of the logarithm (default is natural log).

    Methods:
        transform(X): Apply log scaling.
        inverse_transform(X): Undo the log transformation.
    """
    def __init__(self, column, base=np.e):
        super().__init__()
        self.column_ = column
        self.base_ = base
        self.offset_ = None

    def fit(self, X, y=None):
        """Ensure all data is positive before applying the log by calculating the minimum offset."""
        # If there are values <= 0, we need to shift them by adding an offset.
        # self.offset_ = np.abs(np.min(X, axis=0)) + 1e-9
        pass

    def transform(self, X):
        """Apply logarithmic scaling."""
        X_t = X.copy()
        X_t[self.column_] = np.log(X_t[self.column_].to_numpy())/ np.log(self.base_)
        return X_t

    def inverse_transform(self, X_scaled):
        """Undo the logarithmic transformation."""
        X_t = X_scaled.copy()
        X_t[self.column_] = self.base_ ** X_t[self.column_]
        return X_t 

    def _get_params(self):
        """Return the parameters for saving."""
        return {
            'base': self.base_,
            'column': self.column_
        }

    def _set_params(self, params):
        """Load the parameters from a saved scaler."""
        self.base_ = params['base']

class LowerThreshold(Scaler):
    """
    Applies a lower bound threshold to any specified column.

    Args:
        column (str): Column name.
        lower_bound (float): Minimum allowed value for the specified column.

    Methods:
        transform(X): Clip values below the lower_bound.
        inverse_transform(X): if needed.
    """
    def __init__(self, column, lower_bound):
        super().__init__()
        self.column_ = column
        self.lower_bound_ = lower_bound

    def fit(self, X):
        """Ensure all data is positive before applying the log by calculating the minimum offset."""
        # If there are values <= 0, we need to shift them by adding an offset.
        # self.offset_ = np.abs(np.min(X, axis=0)) + 1e-9
        pass

    def transform(self, X):
        """Apply logarithmic scaling."""
        X_t = X.copy()
        X_t[self.column_] = X_t[self.column_].apply(lambda x: max(x, self.lower_bound_))
        return X_t

    def inverse_transform(self, X_scaled):
        """Undo the logarithmic transformation."""
        raise NotImplementedError("Inverse transform not implemented.")

    def _get_params(self):
        """Return the parameters for saving."""
        return {
            'lower_bound': self.lower_bound_,
            'column': self.column_
        }

    def _set_params(self, params):
        """Load the parameters from a saved scaler."""
        self.lower_bound_ = params['lower_bound']
        self.column_ = params['column']

class UpperThreshold(Scaler):
    """
    Applies an upper bound threshold to any specified column.

    Args:
        column (str): Column name.
        upper_bound (float): Maximum allowed value for the column.

    Methods:
        transform(X): Clip values above the upper_bound.
        inverse_transform(X): if needed.
    """
    def __init__(self, column, upper_bound):
        super().__init__()
        self.column_ = column
        self.upper_bound_ = upper_bound

    def fit(self, X):
        """Ensure all data is positive before applying the log by calculating the minimum offset."""
        # If there are values <= 0, we need to shift them by adding an offset.
        # self.offset_ = np.abs(np.min(X, axis=0)) + 1e-9
        pass

    def transform(self, X):
        """Apply logarithmic scaling."""
        X_t = X.copy()
        X_t[self.column_] = X_t[self.column_].apply(lambda x: min(x, self.upper_bound_))
        return X_t

    def inverse_transform(self, X_scaled):
        """Undo the logarithmic transformation."""
        raise NotImplementedError("Inverse transform not implemented.")

    def _get_params(self):
        """Return the parameters for saving."""
        return {
            'upper_bound': self.upper_bound_,
            'column': self.column_
        }

    def _set_params(self, params):
        """Load the parameters from a saved scaler."""
        self.upper_bound_ = params['upper_bound']
        self.column_ = params['column']

class CategoricalScaler(Scaler):
    """
    Encodes categorical variables as integers.

    Args:
        column (str): Column name.
        possible_values (list, optional): List of possible categories. 

    Methods:
        fit(X): Mappings from categories to integers.
        transform(X): Encode categories to integers.
        inverse_transform(X): Decode integers back to original categories.
    """
    def __init__(self, column, possible_values=None):
        super().__init__()
        self.column_ = column
        self.unique_cats = possible_values

    def fit(self, X):
        """Ensure all data is positive before applying the log by calculating the minimum offset."""
        # If there are values <= 0, we need to shift them by adding an offset.
        # self.offset_ = np.abs(np.min(X, axis=0)) + 1e-9
        if not self.unique_cats:
            self.unique_cats = np.unique(X[self.column_].to_numpy())
        self.cat_converter_ = {value : cat for cat,value in enumerate(self.unique_cats)}
        self.cat_inverter_ = {cat : value for cat,value in enumerate(self.unique_cats)}

    def transform(self, X):
        """convert categories to consecutive integers"""
        X_t = X.copy()
        X_t[self.column_] = X_t[self.column_].apply(lambda x: self.cat_converter_.get(x, None))
        return X_t

    def inverse_transform(self, X_scaled):
        """convert consecutive integers categories"""
        X_t = X_scaled.copy()
        X_t[self.column_] = X_t[self.column_].apply(lambda x: int(self.cat_inverter_.get(x, None)))
        return X_t

    def _get_params(self):
        """Return the parameters for saving."""
        return {
            'cat_converter': self.cat_converter_,
            'cat_inverter': self.cat_inverter_,
            'column': self.column_
        }

    def _set_params(self, params):
        """Load the parameters from a saved scaler."""
        self.cat_converter_ = params['cat_converter']
        self.cat_inverter_ = params['inverter']
        self.column_ = params['column']

# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [4, 6], [3, 7]])

    # MinMax Scaler
    minmax_scaler = MinMaxScaler()
    X_minmax_scaled = minmax_scaler.fit_transform(X)
    minmax_scaler.save_scaler('minmax_scaler.json')

    new_minmax_scaler = MinMaxScaler()
    new_minmax_scaler.load_scaler('minmax_scaler.json')
    X_minmax_loaded_scaled = new_minmax_scaler.transform(X)

    print("Original Data:\n", X)
    print("MinMax Scaled Data:\n", X_minmax_scaled)
    print("Loaded MinMax Scaler Scaled Data:\n", X_minmax_loaded_scaled)

    # Standard Scaler
    standard_scaler = StandardScaler()
    X_standard_scaled = standard_scaler.fit_transform(X)
    standard_scaler.save_scaler('standard_scaler.json')

    new_standard_scaler = StandardScaler()
    new_standard_scaler.load_scaler('standard_scaler.json')
    X_standard_loaded_scaled = new_standard_scaler.transform(X)

    print("Standard Scaled Data:\n", X_standard_scaled)
    print("Loaded Standard Scaler Scaled Data:\n", X_standard_loaded_scaled)