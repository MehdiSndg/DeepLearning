import numpy as np


class NumpyMLP:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42, l2_lambda=0.0):
        """
        Args:
            layer_sizes: list, e.g. [4, 64, 3] for 1 hidden layer
                         or [4, 64, 32, 3] for 2 hidden layers
            learning_rate: float
            seed: int
            l2_lambda: float, L2 regularization strength
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.seed = seed
        self.l2_lambda = l2_lambda
        self.parameters = {}
        self.num_layers = len(layer_sizes) - 1
        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(self.seed)
        for l in range(1, self.num_layers + 1):
            # He initialization for ReLU layers
            self.parameters[f"W{l}"] = (
                np.random.randn(self.layer_sizes[l], self.layer_sizes[l - 1])
                * np.sqrt(2.0 / self.layer_sizes[l - 1])
            )
            self.parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

    @staticmethod
    def _relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def _relu_derivative(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def _softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def _forward(self, X):
        """Forward propagation. X shape: (n_features, m_samples)"""
        cache = {"A0": X}
        A = X
        for l in range(1, self.num_layers + 1):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = W @ A + b
            cache[f"Z{l}"] = Z

            if l == self.num_layers:
                A = self._softmax(Z)
            else:
                A = self._relu(Z)
            cache[f"A{l}"] = A
        return A, cache

    def _compute_cost(self, A, Y):
        """Categorical cross entropy + L2 regularization."""
        m = Y.shape[1]
        log_probs = -np.sum(Y * np.log(A + 1e-8)) / m

        # L2 regularization
        l2_cost = 0
        if self.l2_lambda > 0:
            for l in range(1, self.num_layers + 1):
                l2_cost += np.sum(np.square(self.parameters[f"W{l}"]))
            l2_cost = (self.l2_lambda / (2 * m)) * l2_cost

        return log_probs + l2_cost

    def _backward(self, Y, cache):
        """Backpropagation."""
        m = Y.shape[1]
        grads = {}

        # Output layer
        dZ = cache[f"A{self.num_layers}"] - Y
        for l in range(self.num_layers, 0, -1):
            A_prev = cache[f"A{l - 1}"]
            grads[f"dW{l}"] = (dZ @ A_prev.T) / m
            if self.l2_lambda > 0:
                grads[f"dW{l}"] += (self.l2_lambda / m) * self.parameters[f"W{l}"]
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                dA = self.parameters[f"W{l}"].T @ dZ
                dZ = dA * self._relu_derivative(cache[f"Z{l - 1}"])

        return grads

    def _update_parameters(self, grads):
        for l in range(1, self.num_layers + 1):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, print_every=100):
        """
        Train the model.
        X shape: (m, n_features) -> transposed internally to (n_features, m)
        y shape: (m, n_classes) one-hot encoded -> transposed internally
        """
        X = X_train.T
        Y = y_train.T
        train_losses = []
        val_losses = []

        for i in range(epochs):
            A, cache = self._forward(X)
            cost = self._compute_cost(A, Y)
            grads = self._backward(Y, cache)
            self._update_parameters(grads)
            train_losses.append(cost)

            if X_val is not None and y_val is not None:
                A_val, _ = self._forward(X_val.T)
                val_cost = self._compute_cost(A_val, y_val.T)
                val_losses.append(val_cost)

            if print_every and i % print_every == 0:
                msg = f"Epoch {i}: train_loss={cost:.4f}"
                if val_losses:
                    msg += f", val_loss={val_losses[-1]:.4f}"
                print(msg)

        return train_losses, val_losses

    def predict(self, X):
        """Returns class indices. X shape: (m, n_features)"""
        A, _ = self._forward(X.T)
        return np.argmax(A, axis=0)
