import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class DatasetManipulation:
    # Variant definitions: name -> dict of include flags
    VARIANT_FLAGS = {
        'flow_only':                  {'flow': True},
        'phist_only':                 {'phist': True},
        'endreason_only':             {'endreason': True},
        'ppi_summary_only':           {'ppi_summary': True},
        'ppi_sequence_only':          {'ppi_sequence': True},
        'flow_endreason_ppisummary':  {'flow': True, 'endreason': True, 'ppi_summary': True},
        'all_features':               {'flow': True, 'phist': True, 'endreason': True,
                                       'ppi_summary': True},
    }

    def __init__(self, dataset: pd.DataFrame, encoders_dict: dict):
        self.dataset = dataset
        self.encoders_dict = encoders_dict

        self.flow_features = ['DURATION', 'BYTES', 'BYTES_REV', 'PACKETS', 'PACKETS_REV']
        self.endreason_features = ['FLOW_ENDREASON_IDLE', 'FLOW_ENDREASON_ACTIVE', 'FLOW_ENDREASON_OTHER']
        self.phist_features = [col for col in self.dataset if col.startswith('PHIST_')]
        self.ppi_summary_features = ['PPI_LEN', 'PPI_DURATION', 'PPI_ROUNDTRIPS']
        self.ppi_sequence_features = [col for col in self.dataset if
                                      col.startswith('PPI_') and col not in self.ppi_summary_features]

    def split_dataset(self):
        all_features = (self.flow_features + self.endreason_features + self.phist_features +
                        self.ppi_summary_features + self.ppi_sequence_features)
        y = self.dataset['APP']
        X = self.dataset[all_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def dataset_filter(self, X_train, X_test,
                       incl_flow_feats, incl_phist_feats, incl_endreason_feats,
                       incl_ppi_summary_feats, incl_ppi_sequence_feats):
        cols = []
        if incl_flow_feats:
            cols += self.flow_features
        if incl_phist_feats:
            cols += self.phist_features
        if incl_endreason_feats:
            cols += self.endreason_features
        if incl_ppi_summary_feats:
            cols += self.ppi_summary_features
        if incl_ppi_sequence_feats:
            cols += self.ppi_sequence_features
        return X_train[cols], X_test[cols]

    def scale_tabular_features(self, X_train, X_test, fit=True):
        """Scale flow and phist columns (if present). Endreason and PPI summary left alone."""
        X_train = X_train.copy()
        X_test = X_test.copy()

        flow_present = [c for c in self.flow_features if c in X_train.columns]
        phist_present = [c for c in self.phist_features if c in X_train.columns]

        if flow_present:
            scaler = self.encoders_dict['flow_RobustScaler']
            if fit:
                scaler.fit(X_train[flow_present])
            X_train[flow_present] = scaler.transform(X_train[flow_present])
            X_test[flow_present] = scaler.transform(X_test[flow_present])

        if phist_present:
            scaler = self.encoders_dict['phist_RobustScaler']
            if fit:
                scaler.fit(X_train[phist_present])
            X_train[phist_present] = scaler.transform(X_train[phist_present])
            X_test[phist_present] = scaler.transform(X_test[phist_present])

        return X_train, X_test

    def build_sequence_features(self, X_train, X_test, seq_len=30, fit_scaler=True):
        def stack(df):
            times = df[[f'PPI_TIMES_{i}' for i in range(seq_len)]].values
            sizes = df[[f'PPI_SIZES_{i}' for i in range(seq_len)]].values
            dirs = df[[f'PPI_DIRS_{i}' for i in range(seq_len)]].values
            return np.stack([times, sizes, dirs], axis=2).astype(np.float32)

        seq_X_train = stack(X_train)
        seq_X_test = stack(X_test)

        flat_train = seq_X_train.reshape(-1, 3)
        flat_test = seq_X_test.reshape(-1, 3)

        real_mask_train = flat_train[:, 2] != 0
        real_mask_test = flat_test[:, 2] != 0

        scaler = self.encoders_dict['ppi_scaler']
        if fit_scaler:
            scaler.fit(flat_train[real_mask_train, :2])

        flat_train_scaled = flat_train.copy()
        flat_train_scaled[:, :2] = scaler.transform(flat_train[:, :2])
        flat_train_scaled[~real_mask_train] = 0.0

        flat_test_scaled = flat_test.copy()
        flat_test_scaled[:, :2] = scaler.transform(flat_test[:, :2])
        flat_test_scaled[~real_mask_test] = 0.0

        seq_X_train = flat_train_scaled.reshape(-1, seq_len, 3).astype(np.float32)
        seq_X_test = flat_test_scaled.reshape(-1, seq_len, 3).astype(np.float32)

        return seq_X_train, seq_X_test

    def prepare_all_variants(self, fit=True):
        """
        Build all seven feature variants from a single train/test split.

        Parameters
        ----------
        fit : bool
            True for week 1 (fit scalers on training data).
            False for weeks 2/3/4 (reuse week-1-fit scalers from encoders_dict).

        Returns
        -------
        variants : dict
            Keys are variant names (see VARIANT_FLAGS). Each value is a dict with:
              - 'X_train', 'X_test' : tabular DataFrames (scaled)
              - 'seq_X_train', 'seq_X_test' : (n, 30, 3) numpy arrays, only if
                the variant includes ppi_sequence; otherwise None.
        y_train, y_test : pd.Series
            Shared across all variants.
        """
        X_train_full, X_test_full, y_train, y_test = self.split_dataset()

        variants = {}
        for name, flags in self.VARIANT_FLAGS.items():
            X_train, X_test = self.dataset_filter(
                X_train_full, X_test_full,
                incl_flow_feats=flags.get('flow', False),
                incl_phist_feats=flags.get('phist', False),
                incl_endreason_feats=flags.get('endreason', False),
                incl_ppi_summary_feats=flags.get('ppi_summary', False),
                incl_ppi_sequence_feats=flags.get('ppi_sequence', False),
            )

            # Tabular scaling (no-op if no flow/phist columns are present)
            X_train_scaled, X_test_scaled = self.scale_tabular_features(
                X_train, X_test, fit=fit
            )

            # Sequence array only when ppi_sequence is included
            seq_X_train, seq_X_test = None, None
            if flags.get('ppi_sequence', False):
                seq_X_train, seq_X_test = self.build_sequence_features(
                    X_train, X_test, fit_scaler=fit
                )

            variants[name] = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'seq_X_train': seq_X_train,
                'seq_X_test': seq_X_test,
            }

        return variants, y_train, y_test