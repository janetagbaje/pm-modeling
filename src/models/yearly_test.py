import pandas as pd
from sklearn.metrics import r2_score
import os

def calculate_yearly_r2(validation_save_path, output_dir="../../models/output_maps"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(validation_save_path)
    df['year'] = df['id'].str[:4]

    # List to store R² values for each year
    r2_scores = []

    # Save individual yearly predictions and compute R² scores
    for year, group in df.groupby('year'):
        yearly_file = os.path.join(output_dir, f'predictions_{year}.csv')
        group.to_csv(yearly_file, index=False)
        
        # Compute R² for the current year, applying 10** to 'conc' and 'pred' columns
        if 'conc' in group.columns and 'pred' in group.columns:
            y_true = 10 ** group['conc']
            y_pred = 10 ** group['pred']
            r2 = r2_score(y_true, y_pred)
            r2_scores.append({'year': year, 'r2': r2})
            print(f"R² for {year}: {r2}")

    # DataFrame to store R² scores for each year
    r2_df = pd.DataFrame(r2_scores)

    # Calculate the R² for the entire dataset
    y_true_all = 10 ** df['conc']
    y_pred_all = 10 ** df['pred']
    r2_all = r2_score(y_true_all, y_pred_all)
    print(f"R² for the whole dataset: {r2_all}")
    r2_df = pd.concat([r2_df, pd.DataFrame({'year': ['Whole Dataset'], 'r2': [r2_all]})], ignore_index=True)

    # Calculate the average R² excluding the whole dataset R²
    average_r2 = r2_df[r2_df['year'] != 'Whole Dataset']['r2'].mean()
    average_row = pd.DataFrame({'year': ['Average'], 'r2': [average_r2]})
    
    # Concatenate yearly R² DataFrame with the average row and save to CSV
    r2_df = pd.concat([r2_df, average_row], ignore_index=True)
    r2_csv_path = os.path.join(output_dir, 'r2_scores_by_year.csv')
    r2_df.to_csv(r2_csv_path, index=False)

    print(f"R² scores saved to {r2_csv_path} (including whole dataset and average R²)")
    return r2_csv_path, r2_df