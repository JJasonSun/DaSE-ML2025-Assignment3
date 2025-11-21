import pandas as pd
import numpy as np
from model import Model


class Solution:
    def __init__(self):
        self.model = Model()
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
        self.string_columns = ['protocol_type', 'service', 'flag']
        self.encoders = {}
        
        # Pre-compute indices for faster lookup in forward
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        self.string_col_indices = {name: i for i, name in enumerate(self.feature_names) if name in self.string_columns}

    def _encode_features(self, X_df, fit_mode=False):
        X_df = X_df.copy()

        for col in self.string_columns:
            if col in X_df.columns and X_df[col].dtype == 'object':
                if fit_mode:
                    self.encoders[col] = {val: idx for idx, val in enumerate(X_df[col].unique())}
                X_df[col] = X_df[col].map(self.encoders.get(col, {})).fillna(0)

        X = X_df.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        return X

    def fit(self, X_df, y, learning_rate=0.6, epochs=4):
        X = self._encode_features(X_df, fit_mode=True)
        self.model.fit(X, y, learning_rate, epochs)

    def forward(self, sample: dict) -> dict:
        # Pre-allocate array and cache sample.get
        x_row = np.empty(41, dtype=np.float64)
        sg = sample.get
        
        # Process string columns (indices 1, 2, 3)
        x_row[1] = self.encoders['protocol_type'].get(sg('protocol_type', ''), 0)
        x_row[2] = self.encoders['service'].get(sg('service', ''), 0)
        x_row[3] = self.encoders['flag'].get(sg('flag', ''), 0)
        
        # Process numeric columns
        x_row[0] = sg('duration', 0)
        x_row[4] = sg('src_bytes', 0)
        x_row[5] = sg('dst_bytes', 0)
        x_row[6] = sg('land', 0)
        x_row[7] = sg('wrong_fragment', 0)
        x_row[8] = sg('urgent', 0)
        x_row[9] = sg('hot', 0)
        x_row[10] = sg('num_failed_logins', 0)
        x_row[11] = sg('logged_in', 0)
        x_row[12] = sg('num_compromised', 0)
        x_row[13] = sg('root_shell', 0)
        x_row[14] = sg('su_attempted', 0)
        x_row[15] = sg('num_root', 0)
        x_row[16] = sg('num_file_creations', 0)
        x_row[17] = sg('num_shells', 0)
        x_row[18] = sg('num_access_files', 0)
        x_row[19] = sg('num_outbound_cmds', 0)
        x_row[20] = sg('is_host_login', 0)
        x_row[21] = sg('is_guest_login', 0)
        x_row[22] = sg('count', 0)
        x_row[23] = sg('srv_count', 0)
        x_row[24] = sg('serror_rate', 0)
        x_row[25] = sg('srv_serror_rate', 0)
        x_row[26] = sg('rerror_rate', 0)
        x_row[27] = sg('srv_rerror_rate', 0)
        x_row[28] = sg('same_srv_rate', 0)
        x_row[29] = sg('diff_srv_rate', 0)
        x_row[30] = sg('srv_diff_host_rate', 0)
        x_row[31] = sg('dst_host_count', 0)
        x_row[32] = sg('dst_host_srv_count', 0)
        x_row[33] = sg('dst_host_same_srv_rate', 0)
        x_row[34] = sg('dst_host_diff_srv_rate', 0)
        x_row[35] = sg('dst_host_same_src_port_rate', 0)
        x_row[36] = sg('dst_host_srv_diff_host_rate', 0)
        x_row[37] = sg('dst_host_serror_rate', 0)
        x_row[38] = sg('dst_host_srv_serror_rate', 0)
        x_row[39] = sg('dst_host_rerror_rate', 0)
        x_row[40] = sg('dst_host_srv_rerror_rate', 0)
        
        probability = self.model.predict_proba_single(x_row)
        prediction = int(probability >= 0.5)

        return {
            'prediction': prediction,
            'probability': probability
        }