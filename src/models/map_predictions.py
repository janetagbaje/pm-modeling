import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import h3
import h3pandas
import os

# Define the output directory for saving maps
OUTPUT_MAPS_DIR = "../../models/output_maps"
os.makedirs(OUTPUT_MAPS_DIR, exist_ok=True)

def is_valid_h3(h3_index):
    """Check if an H3 index is valid."""
    try:
        # Check if the index can be converted to a valid parent at the current resolution
        h3.h3_to_parent(h3_index, h3.h3_get_resolution(h3_index))
        return True
    except:
        return False

def plot_predictions_map(inference_save_path, output_dir=OUTPUT_MAPS_DIR):
    # Load prediction data
    df = pd.read_csv(inference_save_path)
    
    # Ensure H3 index is treated as a string
    df['h3_08'] = df['h3_08'].astype(str)

    # Set the H3 index column as the DataFrame's index
    df.set_index('h3_08', inplace=True)

    # Select only the 'pred' column
    df_predictions = df[['pred']]

    # Identify and remove invalid H3 indexes
    invalid_rows = df_predictions[~df_predictions.index.map(is_valid_h3)]
    if not invalid_rows.empty:
        print("Rows with invalid H3 indexes:")
        print(invalid_rows)
    df_predictions = df_predictions[~df_predictions.index.isin(invalid_rows.index)]

    # Convert H3 indices to geometries (leave at resolution 8)
    df_predictions = df_predictions.h3.h3_to_geo_boundary()

    # Apply inverse log transform to 'pred'
    df_predictions['pred'] = 10 ** df_predictions['pred']

    # Plotting the predictions
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot with 'pred' values, using the H3 geometry (no aggregation)
    df_predictions.plot(column='pred', ax=ax, legend=True, cmap='RdPu', antialiased=False)

    # Add title and remove axis
    ax.set_title('SENSEIVER US CONUS PREDICTIONS MAP')
    ax.set_axis_off()

    # Save plot in multiple formats in the specified output directory
    for ext in ['png', 'eps', 'pdf']:
        plt.savefig(os.path.join(output_dir, f"predictions_map.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
