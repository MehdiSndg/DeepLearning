"""
Tüm modellerin metrics.json dosyalarını okur, tek bir karşılaştırma tablosu
(summary_table.csv) ve bar chart (summary_plot.png) üretir.

README raporunun Sonuçlar bölümü bu çıktıları referans alır.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# (model etiketi, metrics.json yolu)
SOURCES = [
    ("Model 1 (LeNet Temel)",    PROJECT_DIR / "model1_lenet_basic"    / "outputs" / "metrics.json"),
    ("Model 2 (LeNet İyileşt.)", PROJECT_DIR / "model2_lenet_improved" / "outputs" / "metrics.json"),
    ("Model 3 (AlexNet)",        PROJECT_DIR / "model3_alexnet"        / "outputs" / "metrics.json"),
    ("Model 4a (AlexNet+SVM)",   PROJECT_DIR / "model4_hybrid"         / "outputs" / "svm_metrics.json"),
    ("Model 4b (AlexNet+RF)",    PROJECT_DIR / "model4_hybrid"         / "outputs" / "rf_metrics.json"),
]

METRIC_KEYS = ["test_accuracy", "precision_macro", "recall_macro", "f1_macro"]


def main():
    rows = []
    for label, path in SOURCES:
        if not path.exists():
            print(f"[UYARI] Bulunamadı: {path} — bu model atlanacak.")
            continue
        with open(path) as f:
            m = json.load(f)
        row = {"model": label}
        for k in METRIC_KEYS:
            row[k] = m.get(k)
        row["training_time_sec"] = m.get("training_time_sec")
        rows.append(row)

    if not rows:
        print("Hiçbir model metrics.json dosyası bulunamadı. Önce eğitimleri çalıştırın.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / "summary_table.csv", index=False)
    print("\n=== Karşılaştırma Tablosu ===")
    print(df.to_string(index=False))

    # Bar chart: test accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["model"], df["test_accuracy"], color="steelblue", edgecolor="black")
    for i, v in enumerate(df["test_accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("CIFAR-10 — Modellerin Test Accuracy Karşılaştırması")
    ax.set_ylim(0, min(1.0, df["test_accuracy"].max() + 0.1))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "summary_plot.png", dpi=150)
    plt.close()

    print(f"\nKaydedildi: {SCRIPT_DIR / 'summary_table.csv'}")
    print(f"Kaydedildi: {SCRIPT_DIR / 'summary_plot.png'}")


if __name__ == "__main__":
    main()
