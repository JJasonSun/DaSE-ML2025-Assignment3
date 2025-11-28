from typing import Any, Callable, Optional, cast
import numpy as np
import math

class TreeNode:
    __slots__ = ('is_leaf', 'value', 'feature_idx', 'threshold', 'threshold_bin', 'left', 'right')
    def __init__(self, is_leaf=False, value=0.0, feature_idx=None, threshold=None, threshold_bin=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.threshold_bin = threshold_bin
        self.left = left
        self.right = right

class XGBoostTree:
    def __init__(self, max_depth=5, lambda_reg=1.0, gamma=0.0, min_child_weight=1.0, n_bins=32, colsample_bytree=1.0):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_bins = n_bins
        self.colsample_bytree = colsample_bytree
        self.root = None

    def fit(self, X_binned, g, h, bin_edges):
        n_samples, n_features = X_binned.shape
        
        if self.colsample_bytree < 1.0:
            feature_indices = np.random.choice(n_features, size=int(n_features * self.colsample_bytree), replace=False)
        else:
            feature_indices = np.arange(n_features)
            
        G_total = np.sum(g)
        H_total = np.sum(h)
        self.root = TreeNode(is_leaf=True, value=self._calc_leaf_weight(G_total, H_total))
        
        nodes = {0: self.root}
        node_assignment = np.zeros(n_samples, dtype=np.int32)
        active_node_ids = [0]
        next_node_id = 1
        
        for depth in range(self.max_depth):
            if not active_node_ids:
                break
            
            valid_indices = np.flatnonzero(np.isin(node_assignment, active_node_ids))
            if len(valid_indices) == 0:
                break
                
            X_active = X_binned[valid_indices]
            g_active = g[valid_indices]
            h_active = h[valid_indices]
            nodes_active = node_assignment[valid_indices]
            
            max_nid = max(active_node_ids)
            lookup = np.full(max_nid + 1, -1, dtype=np.int32)
            lookup[active_node_ids] = np.arange(len(active_node_ids))
            compact_ids = lookup[nodes_active]
            n_active = len(active_node_ids)
            
            best_splits = {} 
            
            node_Gs = np.bincount(compact_ids, weights=g_active, minlength=n_active)
            node_Hs = np.bincount(compact_ids, weights=h_active, minlength=n_active)
            
            for f_idx in feature_indices:
                flat_bins = compact_ids * (self.n_bins + 1) + X_active[:, f_idx]
                minlength = n_active * (self.n_bins + 1)
                
                g_hist = np.bincount(flat_bins, weights=g_active, minlength=minlength).reshape(n_active, self.n_bins + 1)
                h_hist = np.bincount(flat_bins, weights=h_active, minlength=minlength).reshape(n_active, self.n_bins + 1)
                
                G_L = np.cumsum(g_hist, axis=1)
                H_L = np.cumsum(h_hist, axis=1)
                
                G_R = node_Gs[:, None] - G_L
                H_R = node_Hs[:, None] - H_L
                
                mask = (H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)
                
                denom_L = H_L + self.lambda_reg
                denom_R = H_R + self.lambda_reg
                denom_P = node_Hs[:, None] + self.lambda_reg
                
                gains = 0.5 * (np.square(G_L) / denom_L + np.square(G_R) / denom_R - np.square(node_Gs[:, None]) / denom_P) - self.gamma
                gains[~mask] = -np.inf
                
                max_gains = np.max(gains, axis=1)
                best_bins = np.argmax(gains, axis=1)
                
                for i in range(n_active):
                    gain = max_gains[i]
                    if gain > 0:
                        node_id = active_node_ids[i]
                        if node_id not in best_splits or gain > best_splits[node_id][0]:
                            thresh_bin = best_bins[i]
                            edges = bin_edges[f_idx]
                            thresh = edges[thresh_bin] if thresh_bin < len(edges) else edges[-1]
                            best_splits[node_id] = (gain, f_idx, thresh, thresh_bin)

            new_active_nodes = []
            for node_id in active_node_ids:
                if node_id in best_splits:
                    gain, f_idx, thresh, thresh_bin = best_splits[node_id]
                    node = nodes[node_id]
                    node.is_leaf = False
                    node.feature_idx = f_idx
                    node.threshold = thresh
                    node.threshold_bin = thresh_bin
                    
                    left_id = next_node_id
                    right_id = next_node_id + 1
                    next_node_id += 2
                    
                    node.left = TreeNode(is_leaf=True)
                    node.right = TreeNode(is_leaf=True)
                    nodes[left_id] = node.left
                    nodes[right_id] = node.right
                    
                    new_active_nodes.extend([left_id, right_id])
                    
                    cid = lookup[node_id]
                    node_indices_in_active = (compact_ids == cid)
                    original_indices = valid_indices[node_indices_in_active]
                    
                    left_local_mask = X_active[node_indices_in_active, f_idx] <= thresh_bin
                    left_indices = original_indices[left_local_mask]
                    right_indices = original_indices[~left_local_mask]
                    
                    node_assignment[left_indices] = left_id
                    node_assignment[right_indices] = right_id
                    
                    G_L = np.sum(g[left_indices])
                    H_L = np.sum(h[left_indices])
                    node.left.value = self._calc_leaf_weight(G_L, H_L)
                    
                    G_R = np.sum(g[right_indices])
                    H_R = np.sum(h[right_indices])
                    node.right.value = self._calc_leaf_weight(G_R, H_R)
            
            active_node_ids = new_active_nodes

    def _calc_leaf_weight(self, G, H):
        return -G / (H + self.lambda_reg)

    def predict_binned(self, X_binned):
        n_samples = X_binned.shape[0]
        predictions = np.zeros(n_samples)
        self._predict_binned_recursive(X_binned, np.arange(n_samples), self.root, predictions)
        return predictions

    def _predict_binned_recursive(self, X_binned, indices, node, predictions):
        if node.is_leaf:
            predictions[indices] = node.value
            return
        mask = X_binned[indices, node.feature_idx] <= node.threshold_bin
        left_indices = indices[mask]
        right_indices = indices[~mask]
        if len(left_indices) > 0:
            self._predict_binned_recursive(X_binned, left_indices, node.left, predictions)
        if len(right_indices) > 0:
            self._predict_binned_recursive(X_binned, right_indices, node.right, predictions)

class Model:
    def __init__(self, n_features=41, feature_names=None, random_state=42):
        self.n_features = n_features
        self.feature_names = feature_names
        self.encoders = {}
        self.trees = []
        self.bin_edges = []
        self.n_bins = 32
        self.max_depth = 4
        self.learning_rate = 0.3
        self.lambda_reg = 1.0
        self.gamma = 0.1
        self.min_child_weight = 1.0
        self.colsample_bytree = 0.8
        self.subsample = 0.8
        self.base_score = 0.0
        self.fast_predict: Optional[Callable[[Any], float]] = None
        
    def _sigmoid(self, z):
        z = np.clip(z, -100, 100) 
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, learning_rate=0.3, epochs=10):
        self.learning_rate = learning_rate
        
        # Optimization: If single epoch, subsample upfront to save binning cost
        if epochs == 1 and self.subsample < 1.0:
            n_samples = len(y)
            n_sub = int(n_samples * self.subsample)
            if n_sub > 0:
                idx = np.random.choice(n_samples, size=n_sub, replace=False)
                X = X[idx]
                y = y[idx]
                # Disable loop subsampling since we already subsampled
                self.subsample = 1.0

        preds = np.full(y.shape, self.base_score)
        
        self.bin_edges = []
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        for i in range(X.shape[1]):
            lo, hi = col_min[i], col_max[i]
            if hi > lo:
                edges = np.linspace(lo, hi, self.n_bins + 1)[1:-1]
            else:
                edges = np.array([lo])
            self.bin_edges.append(edges)
            X_binned[:, i] = np.searchsorted(edges, X[:, i])

        for epoch in range(epochs):
            p = self._sigmoid(preds)
            g = p - y
            h = p * (1 - p)
            
            if self.subsample < 1.0:
                abs_g = np.abs(g)
                sorted_idx = np.argsort(abs_g)[::-1]
                n_samples = len(y)
                
                # GOSS: Keep top gradients and sample from the rest
                # Allocate 1/3 of budget to top gradients, 2/3 to random
                top_n = int(n_samples * self.subsample * 0.33)
                rand_n = int(n_samples * self.subsample * 0.67)
                
                if top_n + rand_n > n_samples:
                    top_n = n_samples
                    rand_n = 0
                
                top_idx = sorted_idx[:top_n]
                rest_idx = sorted_idx[top_n:]
                
                if rand_n > len(rest_idx):
                    rand_n = len(rest_idx)
                    
                if rand_n > 0:
                    rand_idx = np.random.choice(rest_idx, size=rand_n, replace=False)
                    idx = np.concatenate([top_idx, rand_idx])
                    weight_multiplier = (len(rest_idx) / rand_n)
                    X_sample = X_binned[idx]
                    g_sample = g[idx].copy()
                    h_sample = h[idx].copy()
                    # Scale gradients of random samples to correct distribution
                    g_sample[top_n:] *= weight_multiplier
                    h_sample[top_n:] *= weight_multiplier
                else:
                    idx = top_idx
                    X_sample = X_binned[idx]
                    g_sample = g[idx]
                    h_sample = h[idx]
            else:
                X_sample = X_binned
                g_sample = g
                h_sample = h

            tree = XGBoostTree(
                max_depth=self.max_depth,
                lambda_reg=self.lambda_reg,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                n_bins=self.n_bins,
                colsample_bytree=self.colsample_bytree
            )
            tree.fit(X_sample, g_sample, h_sample, self.bin_edges)
            self.trees.append(tree)
            
            update = tree.predict_binned(X_binned)
            preds += self.learning_rate * update
            
        self._compile_trees()

    def _compile_trees(self):
        used_features = set()
        feature_names = self.feature_names
        
        def collect_features(node):
            if node.is_leaf: return
            if feature_names:
                used_features.add(feature_names[node.feature_idx])
            collect_features(node.left)
            collect_features(node.right)
            
        for tree in self.trees:
            collect_features(tree.root)
            
        code_lines = []
        code_lines.append('def fast_predict(x):')
        
        for f in used_features:
            code_lines.append(f'    var_{f} = x[\'{f}\']')
            
        code_lines.append(f'    val = {self.base_score}')
        
        namespace: dict[str, object] = {'math': math}
        
        for i, tree in enumerate(self.trees):
            def visit(node, indent):
                if node.is_leaf:
                    val = self.learning_rate * node.value
                    return [f'{indent}val += {val}']
                
                lines = []
                if feature_names:
                    fname = feature_names[node.feature_idx]
                    var_name = f'var_{fname}'
                    if fname in self.encoders:
                        encoder = self.encoders[fname]
                        valid_values = {k for k, v in encoder.items() if v <= node.threshold}
                        lines.append(f'{indent}if {var_name} in {repr(valid_values)}:')
                    else:
                        lines.append(f'{indent}if {var_name} <= {node.threshold}:')
                else:
                    lines.append(f'{indent}if x[{node.feature_idx}] <= {node.threshold}:')
                    
                lines.extend(visit(node.left, indent + '    '))
                lines.append(f'{indent}else:')
                lines.extend(visit(node.right, indent + '    '))
                return lines

            code_lines.extend(visit(tree.root, '    '))
            
        code_lines.append('    if val < -100: return 0.0')
        code_lines.append('    if val > 100: return 1.0')
        code_lines.append('    return 1.0 / (1.0 + math.exp(-val))')
        
        source = '\n'.join(code_lines)
        exec(source, namespace)
        self.fast_predict = cast(Callable[[Any], float], namespace['fast_predict'])
        # Optimization: Bypass method call overhead
        self.predict_proba_single = self.fast_predict

    def predict_proba_single(self, x):
        if self.fast_predict is None:
            raise RuntimeError('Model has not been compiled')
        return self.fast_predict(x)
    
    def predict_proba(self, X):
        if X.shape[0] == 1:
            return np.array([self.predict_proba_single(X[0])])
        preds = np.full(X.shape[0], self.base_score)
        return self._sigmoid(preds)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
