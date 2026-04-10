import ast
import pandas as pd
import numpy as np

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