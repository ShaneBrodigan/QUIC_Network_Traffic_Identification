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

    def perform_encoding(self, scalers: dict, fit: bool):
        cols_to_label_encode = [col for col in self.dataframe.columns.tolist() if pd.api.types.is_string_dtype(self.dataframe[col])]

        ppi_dir_cols = [col for col in self.dataframe.columns.tolist() if col.startswith('PPI_DIRS')]
        ppi_dir_df = self.dataframe[ppi_dir_cols]

        bool_cols = [col for col in self.dataframe.columns.tolist() if self.dataframe[col].dtype in ['bool']]
        bool_df = self.dataframe[bool_cols]

        numeric_cols = [col for col in self.dataframe.columns.tolist()
                        if self.dataframe[col].dtype in ['int64', 'float64']
                        and not col.startswith('PPI_DIRS')]
        numeric_df = self.dataframe[numeric_cols]

        print('Label Encoding...')
        encoded_df = self.encode(cols_to_label_encode, scalers['label_encoder'], fit)

        print('Merging final df')
        final_df = pd.concat([encoded_df, numeric_df, ppi_dir_df, bool_df], axis=1)

        self.dataframe = final_df

    def encode(self, col_names: list[str], label_encoders: dict, fit: bool) -> pd.DataFrame:
        scaled_df = pd.DataFrame()
        for col in col_names:
            if fit:
                label_encoders[col] = LabelEncoder()
                scaled_df[col] = label_encoders[col].fit_transform(self.dataframe[col])
            else:
                scaled_df[col] = label_encoders[col].transform(self.dataframe[col])
        return scaled_df

    def get_tabular_dataset(self) -> pd.DataFrame:
        return self.dataframe


def robust_scale(df: pd.DataFrame, scaler: RobustScaler, fit: bool) -> pd.DataFrame:
    cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64'] and not c.startswith('PPI_DIRS')]
    df = df.copy()
    if fit:
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df
