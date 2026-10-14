import h3, os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import inspect
import os


def h3_to_polygon(h3_index):
    """Convert H3 index to a Shapely polygon for visualization."""
    boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
    return Polygon(boundary)


def generate_r2_mae_maps(
    validation_data, output_dir="../../models/output_maps", output_res=2, map_kwargs={}
):
    os.makedirs(output_dir, exist_ok=True)

    if type(validation_data) == str:
        sensor_data = pd.read_csv(validation_data)
    elif type(validation_data) == pd.DataFrame:
        sensor_data = validation_data

    # Define resolution level for coarser H3 aggregation
    sensor_data["h3_02"] = sensor_data["h3_08"].apply(
        lambda x: h3.h3_to_parent(x, output_res)
    )

    # Apply inverse log transform to 'conc' and 'pred' columns
    sensor_data["conc"] = 10 ** sensor_data["conc"]
    sensor_data["pred"] = 10 ** sensor_data["pred"]

    # Calculate metrics for each aggregated H3 index
    sensor_agg = (
        sensor_data.groupby("h3_02")
        .apply(
            lambda group: pd.Series(
                {
                    "r2": r2_score(group["conc"], group["pred"]),
                    "mae": mean_absolute_error(group["conc"], group["pred"]),
                    "rmse": np.sqrt(mean_squared_error(group["conc"], group["pred"])),
                    "ndata": len(group),
                    "hex_mean_data": group["conc"].mean(),
                    "hex_mean_pred": group["pred"].mean(),
                    "hex_mae": np.abs(group["conc"].mean() - group["pred"].mean()),
                }
            )
        )
        .reset_index()
    )

    # Convert aggregated H3 indices to geometries
    sensor_agg["geometry"] = sensor_agg["h3_02"].apply(h3_to_polygon)
    sensor_agg_gdf = gpd.GeoDataFrame(sensor_agg, geometry="geometry")

    # Initialize dictionary to hold paths of saved images
    image_paths = {}

    # Fields to visualize
    fields = ["mae", "r2", "ndata", "hex_mean_data", "hex_mean_pred", "hex_mae"]
    for field in fields:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        sensor_agg_gdf.plot(
            column=field, ax=ax, legend=False, cmap="RdPu", antialiased=False
        )
        ax.set_axis_off()

        # Add colorbar with custom placement
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        if f"vmin_{field}" not in map_kwargs.keys():
            map_kwargs["vmin"] = sensor_agg_gdf[field].min()
        else:
            map_kwargs["vmin"] = map_kwargs[f"vmin_{field}"]

        if f"vmax_{field}" not in map_kwargs.keys():
            map_kwargs["vmax"] = sensor_agg_gdf[field].max()
        else:
            map_kwargs["vmax"] = map_kwargs[f"vmax_{field}"]

        sm = plt.cm.ScalarMappable(cmap="RdPu", norm=normalize(**map_kwargs))
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(f"{field}", labelpad=10)

        # Save in multiple formats
        for ext in ["png", "eps", "pdf"]:
            image_path = os.path.join(output_dir, f"{field}_map.{ext}")
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            image_paths[f"{field}_{ext}"] = image_path
        plt.close(fig)

    return image_paths, sensor_agg_gdf


def normalize(**kwargs):
    # Get the parameter names of the function
    valid_params = inspect.signature(plt.Normalize).parameters

    # Filter kwargs to only include valid parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return plt.Normalize(**filtered_kwargs)
