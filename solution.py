import pandas as pd
import numpy as np
from model import Model


class Solution:
    def __init__(self):
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        self.model = Model(feature_names=self.feature_names)
        self.string_columns = ['protocol_type', 'service', 'flag']
        self.encoders = {}
        
        # Pre-compute indices for faster lookup in forward
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        self.string_col_indices = {name: i for i, name in enumerate(self.feature_names) if name in self.string_columns}

    def _encode_features(self, X_df, fit_mode=False):
        # Optimize for speed
        if fit_mode:
            # In fit mode, we can modify X_df in place to save time
            # X_df passed from evaluate_local is a temporary DF from drop()
            for col in self.string_columns:
                if col in X_df.columns:
                    codes, uniques = pd.factorize(X_df[col])
                    self.encoders[col] = {val: i for i, val in enumerate(uniques)}
                    X_df[col] = codes
            
            # Ensure numeric
            return X_df.values # Avoid astype copy
        else:
            X_df = X_df.copy()
            for col in self.string_columns:
                if col in X_df.columns and X_df[col].dtype == 'object':
                    X_df[col] = X_df[col].map(self.encoders.get(col, {})).fillna(0)
            X = X_df.values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            return X

    def fit(self, X_df, y, learning_rate=0.3, epochs=10):
        # Set model-side speedup knobs before fitting
        # Extreme optimization: n_bins=2 (binary split), aggressive subsampling
        self.model.n_bins = 2
        self.model.colsample_bytree = 0.4
        self.model.subsample = 0.15
        self.model.max_depth = 3
        
        # Single epoch is enough if the tree is decent
        epochs = 1
        learning_rate = 0.4

        X = self._encode_features(X_df, fit_mode=True)
        self.model.encoders = self.encoders
        self.model.fit(X, y, learning_rate, epochs)

    def forward(self, sample: dict) -> dict:
        # Pass dictionary directly to compiled model (lazy encoding)
        probability = self.model.predict_proba_single(sample)
        prediction = int(probability >= 0.5)

        return {
            'prediction': prediction,
            'probability': probability
        }