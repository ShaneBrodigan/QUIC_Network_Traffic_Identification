import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler

class Feature_Engineering():

    def __init__(self, dataframe):
        self.dataframe = dataframe

    # Parses the PPI column into PPI_times, PPI_DIRS and PPI_SIZES and appends to returned dataframe
    def parse_col(self, target_col: str, output_cols: list[str]) -> pd.DataFrame:
        parsed = self.dataframe[target_col].apply(lambda x: ast.literal_eval(x)).tolist()
        self.dataframe = self.append_to_df(parsed, target_col, output_cols)
        return self.dataframe

    def append_to_df(self, parsed_target, target_col: str, output_cols: list[str]) -> pd.DataFrame:
        parsed_cols = pd.DataFrame()

        if isinstance(parsed_target[0][0], list):
            # PPI - nested list, extract each sub-list into its own column
            for counter, output_col in enumerate(output_cols):
                parsed_cols[output_col] = [row[counter] for row in parsed_target]
        else:
            # PHIST - flat list, assign directly as single column
            parsed_cols[output_cols[0]] = parsed_target

        df = self.dataframe.drop(columns=target_col)
        df = df.merge(parsed_cols, left_index=True, right_index=True)
        return df

    # Parses columns of type list into individual columns per indices, applies padding where lists lengths vary
    def parse_col_lists(self, padding_length: int = 30):
        list_cols = [col for col in self.dataframe.columns if isinstance(self.dataframe[col].iloc[0], list)]
        list_cols_df = self.dataframe[list_cols].copy()
        new_cols = {}

        for col in list_cols:
            padded = list_cols_df[col].apply(
                lambda x: x + [0] * (padding_length - len(x)) if len(x) < padding_length else x[:padding_length])
            arr = np.array(padded.tolist())

            for i in range(arr.shape[1]):
                new_cols[f'{col}_{i}'] = arr[:, i]

        self.dataframe = self.dataframe.drop(columns=list_cols)
        self.dataframe = pd.concat([self.dataframe, pd.DataFrame(new_cols)], axis=1)
        return self.dataframe

    def perform_encode_and_scaling(self, scalers: dict):
        cols_to_robustscale = [col for col in self.dataframe.columns.tolist() if self.dataframe[col].dtype in ['int64', 'float64']]
        cols_to_robustscale = [col for col in cols_to_robustscale if
                               not col.startswith('PPI_DIRS')]  # Removing PPI_DIR cols as they are already -1, 0 or 1

        cols_to_label_encode = [col for col in self.dataframe.columns.tolist() if self.dataframe[col].dtype in ['str']]

        ppi_dir_cols = [col for col in self.dataframe.columns.tolist() if col.startswith('PPI_DIRS')]
        ppi_dir_df = self.dataframe[ppi_dir_cols]

        bool_cols = [col for col in self.dataframe.columns.tolist() if self.dataframe[col].dtype in ['bool']]
        bool_df = self.dataframe[bool_cols]

        print('Label Encoding...')
        encoded_df = self.encode(cols_to_label_encode, scalers['label_encoder'])

        print('Robust Scaling...')
        scaled_array = scalers['RobustScaler'].fit_transform(self.dataframe[cols_to_robustscale])
        scaled_df = pd.DataFrame(scaled_array, columns=cols_to_robustscale)

        print('Merging final df')
        final_df = pd.concat([encoded_df, scaled_df, ppi_dir_df, bool_df], axis=1)

        self.dataframe = final_df

    def encode(self, col_names: list[str], scaler) -> pd.DataFrame:
        scaled_df = pd.DataFrame()
        for col in col_names:
            scaled = scaler.fit_transform(self.dataframe[col])
            scaled_df[col] = scaled
        return scaled_df

    def get_tabular_dataset(self) -> pd.DataFrame:
        return self.dataframe

    def get_ppi_sequential_only(self):
        ppi_cols = (
                [f'PPI_TIMES_{i}' for i in range(30)] +
                [f'PPI_DIRS_{i}' for i in range(30)] +
                [f'PPI_SIZES_{i}' for i in range(30)]
        )
        return self.dataframe[['APP', 'CATEGORY'] + ppi_cols]
