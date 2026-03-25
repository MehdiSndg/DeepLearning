from sklearn.neural_network import MLPClassifier
import numpy as np


class SklearnMLP:
    def __init__(self, hidden_layer_sizes=(64,), learning_rate=0.01, seed=42,
                 l2_lambda=0.0):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="sgd",
            alpha=l2_lambda,
            learning_rate_init=learning_rate,
            max_iter=1000,
            random_state=seed,
            verbose=False,
        )
        self._hidden = hidden_layer_sizes
        self._lr = learning_rate
        self._seed = seed
        self._l2 = l2_lambda

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, print_every=100):
        if y_train.ndim > 1:
            y_train = np.argmax(y_train, axis=1)

        self.model.max_iter = epochs
        self.model.batch_size = X_train.shape[0]  # full batch like our NumPy model
        self.model.fit(X_train, y_train)

        train_losses = self.model.loss_curve_
        val_losses = []
        return train_losses, val_losses

    def predict(self, X):
        return self.model.predict(X)
