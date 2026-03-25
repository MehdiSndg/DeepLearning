import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]


def load_data(path="data/Dry_Bean_Dataset.xlsx"):
    df = pd.read_excel(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    X = df.iloc[:, :-1].to_numpy().astype(np.float64)
    y = df["Class"].map(label_map).to_numpy()
    return X, y


def one_hot_encode(y, num_classes=7):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def split_data(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=seed, stratify=y
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_val, X_test, mean, std
