from multiprocessing import Pool
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

with open('config.json') as f:
    config = json.load(f)

COLS_TO_DROP = config['cols_to_drop']

def convert_to_parquet(dataset_root):
    multiprocess_queue = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                multiprocess_queue.append(file_path)

    with Pool(processes=4) as pool:
        pool.map(convert_file, multiprocess_queue)

def convert_file(csv_file_path: str):
    parquet_path = csv_file_path.replace('.csv', '.parquet')

    writer = None
    for chunk in pd.read_csv(csv_file_path, chunksize=100_000, dtype={'QUIC_USERAGENT': 'str'}):
        chunk.drop(columns=[c for c in COLS_TO_DROP if c in chunk.columns], inplace=True)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    os.remove(csv_file_path)
    print(f'Done: {parquet_path}')

if __name__ == '__main__':
    dataset_root = './dataset'
    convert_to_parquet(dataset_root)