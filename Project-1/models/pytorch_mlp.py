import torch
import torch.nn as nn
import numpy as np


class PytorchMLP:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42, l2_lambda=0.0):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.seed = seed
        self.l2_lambda = l2_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers).to(self.device)

        # He initialization
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, weight_decay=l2_lambda
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, print_every=100):
        if y_train.ndim > 1:
            y_train = np.argmax(y_train, axis=1)

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)

        X_v, y_v = None, None
        if X_val is not None and y_val is not None:
            if y_val.ndim > 1:
                y_val = np.argmax(y_val, axis=1)
            X_v = torch.FloatTensor(X_val).to(self.device)
            y_v = torch.LongTensor(y_val).to(self.device)

        train_losses = []
        val_losses = []

        self.model.train()
        for i in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = self.criterion(outputs, y_t)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

            if X_v is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_v)
                    val_loss = self.criterion(val_out, y_v)
                    val_losses.append(val_loss.item())
                self.model.train()

            if print_every and i % print_every == 0:
                msg = f"Epoch {i}: train_loss={loss.item():.4f}"
                if val_losses:
                    msg += f", val_loss={val_losses[-1]:.4f}"
                print(msg)

        return train_losses, val_losses

    def predict(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
