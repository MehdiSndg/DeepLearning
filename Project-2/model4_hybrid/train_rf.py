"""
Model 4b — Hibrit: AlexNet özellikleri + Random Forest

features/*.npy dosyalarını okur ve Random Forest sınıflandırıcısını eğitir.
RF ölçeklemeye duyarsız olduğundan StandardScaler kullanılmaz.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FEAT_DIR = SCRIPT_DIR / "features"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_ESTIMATORS = 200
MAX_DEPTH = None
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_features():
    X_train = np.load(FEAT_DIR / "X_train.npy")
    y_train = np.load(FEAT_DIR / "y_train.npy")
    X_test = np.load(FEAT_DIR / "X_test.npy")
    y_test = np.load(FEAT_DIR / "y_test.npy")
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(cm, path, title):
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    X_train, y_train, X_test, y_test = load_features()
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test : X={X_test.shape}, y={y_test.shape}")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1,
        random_state=SEED,
    )

    print(f"\nRandom Forest eğitiliyor (n_estimators={N_ESTIMATORS})...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Eğitim süresi: {train_time:.1f} sn")

    print("Test değerlendirmesi...")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - t0
    print(f"Tahmin süresi: {pred_time:.1f} sn")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "model": "Model 4b — AlexNet özellikleri + Random Forest",
        "dataset": "CIFAR-10 (AlexNet feature embeddings, 4096-d)",
        "classifier": f"RandomForestClassifier(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_dim": int(X_train.shape[1]),
        "training_time_sec": round(train_time, 2),
        "prediction_time_sec": round(pred_time, 2),
        "test_accuracy": round(acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
    }

    with open(OUT_DIR / "rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    (OUT_DIR / "rf_classification_report.txt").write_text(report)

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUT_DIR / "rf_confusion_matrix.csv"
    )
    plot_confusion_matrix(cm, OUT_DIR / "rf_confusion_matrix.png",
                          "Model 4b (AlexNet + Random Forest) — Karmaşıklık Matrisi")

    print("\n=== ÖZET ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
