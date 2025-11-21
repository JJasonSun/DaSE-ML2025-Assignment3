import numpy as np
import math

class TreeNode:
    def __init__(self, is_leaf=False, value=0.0, feature_idx=None, threshold=None, threshold_bin=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.threshold_bin = threshold_bin
        self.left = left
        self.right = right
        self.my_idx = 0 # Helper for flattening

class XGBoostTree:
    def __init__(self, max_depth=5, lambda_reg=1.0, gamma=0.0, min_child_weight=1.0, n_bins=32, colsample_bytree=0.8):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_bins = n_bins
        self.colsample_bytree = colsample_bytree
        self.root = None
        self.bin_edges = None

    def fit(self, X_binned, g, h, bin_edges):
        self.bin_edges = bin_edges
        n_features = X_binned.shape[1]
        if self.colsample_bytree < 1.0:
            self.feature_indices = np.random.choice(n_features, size=int(n_features * self.colsample_bytree), replace=False)
        else:
            self.feature_indices = np.arange(n_features)
            
        self.root = self._build_tree(X_binned, g, h, depth=0)

    def _build_tree(self, X_binned, g, h, depth):
        G = np.sum(g)
        H = np.sum(h)
        
        if depth >= self.max_depth or H < self.min_child_weight:
            return TreeNode(is_leaf=True, value=self._calc_leaf_weight(G, H))

        best_gain = 0.0
        best_feature_idx = None
        best_threshold_bin = None
        
        for feature_idx in self.feature_indices:
            g_hist = np.bincount(X_binned[:, feature_idx], weights=g, minlength=self.n_bins + 1)
            h_hist = np.bincount(X_binned[:, feature_idx], weights=h, minlength=self.n_bins + 1)
            
            G_L = np.cumsum(g_hist)
            H_L = np.cumsum(h_hist)
            
            G_R = G - G_L
            H_R = H - H_L
            
            mask = (H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)
            
            if not np.any(mask):
                continue
                
            current_gain = 0.5 * (
                (G_L[mask]**2) / (H_L[mask] + self.lambda_reg) + 
                (G_R[mask]**2) / (H_R[mask] + self.lambda_reg) - 
                (G**2) / (H + self.lambda_reg)
            ) - self.gamma
            
            if len(current_gain) == 0:
                continue
                
            max_gain_idx = np.argmax(current_gain)
            max_gain = current_gain[max_gain_idx]
            
            if max_gain > best_gain:
                best_gain = max_gain
                best_feature_idx = feature_idx
                valid_indices = np.where(mask)[0]
                best_threshold_bin = valid_indices[max_gain_idx]

        if best_gain > 0:
            left_mask = X_binned[:, best_feature_idx] <= best_threshold_bin
            right_mask = ~left_mask
            
            edges = self.bin_edges[best_feature_idx]
            if best_threshold_bin < len(edges):
                float_threshold = edges[best_threshold_bin]
            else:
                float_threshold = edges[-1]

            left_child = self._build_tree(X_binned[left_mask], g[left_mask], h[left_mask], depth + 1)
            right_child = self._build_tree(X_binned[right_mask], g[right_mask], h[right_mask], depth + 1)
            
            return TreeNode(is_leaf=False, feature_idx=best_feature_idx, threshold=float_threshold, threshold_bin=best_threshold_bin, left=left_child, right=right_child)
        else:
            return TreeNode(is_leaf=True, value=self._calc_leaf_weight(G, H))

    def _calc_leaf_weight(self, G, H):
        return -G / (H + self.lambda_reg)

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        self._predict_recursive(X, np.arange(n_samples), self.root, predictions)
        return predictions

    def predict_single(self, x):
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _predict_recursive(self, X, indices, node, predictions):
        if node.is_leaf:
            predictions[indices] = node.value
            return

        mask = X[indices, node.feature_idx] <= node.threshold
        left_indices = indices[mask]
        right_indices = indices[~mask]
        
        if len(left_indices) > 0:
            self._predict_recursive(X, left_indices, node.left, predictions)
        if len(right_indices) > 0:
            self._predict_recursive(X, right_indices, node.right, predictions)

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
        self.n_bins = 8
        self.max_depth = 3
        self.learning_rate = 1.0
        self.lambda_reg = 1.0
        self.gamma = 0.1
        self.min_child_weight = 1.0
        self.colsample_bytree = 0.8
        self.subsample = 0.8
        self.base_score = 0.0
        self.fast_predict = None
        
    def _sigmoid(self, z):
        z = np.clip(z, -100, 100) 
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, learning_rate=0.6, epochs=4):
        self.learning_rate = learning_rate
        preds = np.full(y.shape, self.base_score)
        
        self.bin_edges = []
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        
        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            if len(unique_vals) <= self.n_bins:
                edges = np.sort(unique_vals)
                self.bin_edges.append(edges)
                X_binned[:, i] = np.searchsorted(edges, X[:, i])
            else:
                percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
                edges = np.percentile(X[:, i], percentiles)
                edges = np.unique(edges)
                self.bin_edges.append(edges)
                X_binned[:, i] = np.searchsorted(edges, X[:, i])

        for epoch in range(epochs):
            p = self._sigmoid(preds)
            g = p - y
            h = p * (1 - p)
            
            if self.subsample < 1.0:
                n_samples = X.shape[0]
                idx = np.random.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
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
        code_lines = []
        code_lines.append("def fast_predict(x):")
        code_lines.append(f"    val = {self.base_score}")
        
        # Helper to handle feature access
        use_dict = self.feature_names is not None
        
        for i, tree in enumerate(self.trees):
            def visit(node, indent):
                if node.is_leaf:
                    # Bake learning rate into the value
                    val = self.learning_rate * node.value
                    return [f"{indent}val += {val}"]
                
                lines = []
                if use_dict:
                    fname = self.feature_names[node.feature_idx]
                    if fname in self.encoders:
                        cond = f"encoders['{fname}'].get(x['{fname}'], 0) <= {node.threshold}"
                    else:
                        cond = f"x['{fname}'] <= {node.threshold}"
                    
                    lines.append(f"{indent}if {cond}:")
                    lines.extend(visit(node.left, indent + "    "))
                    lines.append(f"{indent}else:")
                    lines.extend(visit(node.right, indent + "    "))
                else:
                    lines.append(f"{indent}if x[{node.feature_idx}] <= {node.threshold}:")
                    lines.extend(visit(node.left, indent + "    "))
                    lines.append(f"{indent}else:")
                    lines.extend(visit(node.right, indent + "    "))
                return lines

            code_lines.extend(visit(tree.root, "    "))
            
        code_lines.append("    return val")
        
        source = "\n".join(code_lines)
        namespace = {'encoders': self.encoders}
        exec(source, namespace)
        self.fast_predict = namespace['fast_predict']

    def predict_proba_single(self, x):
        """Optimized single sample prediction using compiled code"""
        pred = self.fast_predict(x)
        
        # Inline sigmoid with math.exp for speed
        if pred < -100: return 0.0
        if pred > 100: return 1.0
        return 1.0 / (1.0 + math.exp(-pred))
    
    def predict_proba(self, X):
        if X.shape[0] == 1:
            return np.array([self.predict_proba_single(X[0])])
        
        preds = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return self._sigmoid(preds)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)