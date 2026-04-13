import pandas as pd
import os

os.makedirs('./tabular_dataset', exist_ok=True)

def write_out_tabular(current_dataframe: pd.DataFrame, file_name):
    file_name = f'tabular_{file_name}'
    tabular_dataset_path = f'./tabular_dataset/{file_name}'
    print(f'TABULAR_DATASET_PATH: {tabular_dataset_path}')
    current_dataframe.to_parquet(tabular_dataset_path)
    print(f'{file_name}: {current_dataframe.shape} saved to {tabular_dataset_path}')