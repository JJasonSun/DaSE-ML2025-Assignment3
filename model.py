import numpy as np


class Model:
    def __init__(self, n_features=41, random_state=42):
        np.random.seed(random_state)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, learning_rate=0.1, epochs=200):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)