"""
Model 3 — AlexNet (pretrained) fine-tune scripti

Veri seti : CIFAR-10 (32x32 -> 224x224'e resize edilir)
Mimari    : torchvision.models.alexnet, pretrained=True
Loss      : CrossEntropyLoss
Optimizer : Adam (lr=1e-4)        ← pretrained ağırlıkları bozmamak için düşük
Batch     : 128
Epoch     : 20
Seed      : 42

Not: Eğitim bittiğinde model ağırlıkları 'checkpoint.pth' olarak kaydedilir
ve Model 4 (hibrit) bu checkpoint'i feature extractor olarak kullanır.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_alexnet

# ───────────────────────── Hiperparametreler ─────────────────────────
SEED = 42
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 20
NUM_CLASSES = 10
PRETRAINED = True
IMG_SIZE = 224

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
CKPT_PATH = SCRIPT_DIR / "checkpoint.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders():
    # AlexNet ImageNet pretrained ağırlıkları için uygun normalizasyon
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),     # 32 -> 224
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return total_loss / total, correct / total, np.concatenate(all_preds), np.concatenate(all_labels)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def plot_curve(history, ycols, ylabel, path):
    plt.figure(figsize=(7, 4.5))
    for c in ycols:
        plt.plot(history["epoch"], history[c], marker="o", label=c)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Model 3 (AlexNet) — {ylabel}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, path):
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Model 3 — Karmaşıklık Matrisi (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    print(f"Cihaz: {DEVICE}")

    train_loader, test_loader = get_loaders()
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_alexnet(num_classes=NUM_CLASSES, pretrained=PRETRAINED).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametre sayısı: {n_params:,} | pretrained={PRETRAINED}")

    history = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion)
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "test_loss": te_loss, "test_acc": te_acc,
        })
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")
    elapsed = time.time() - t0
    print(f"Toplam eğitim süresi: {elapsed:.1f} sn")

    # Checkpoint kaydet (Model 4 feature extraction için gerekli)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Checkpoint kaydedildi: {CKPT_PATH}")

    final_loss, final_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "model": "Model 3 — AlexNet (pretrained, fine-tuned)",
        "dataset": "CIFAR-10 (resize 224)",
        "device": str(DEVICE),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LR,
        "pretrained": PRETRAINED,
        "num_parameters": int(n_params),
        "training_time_sec": round(elapsed, 2),
        "test_loss": round(final_loss, 4),
        "test_accuracy": round(final_acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
    }

    df = pd.DataFrame(history)
    df.to_csv(OUT_DIR / "training_log.csv", index=False)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT_DIR / "confusion_matrix.csv")

    plot_curve(df, ["train_loss", "test_loss"], "Loss", OUT_DIR / "loss_curve.png")
    plot_curve(df, ["train_acc", "test_acc"], "Accuracy", OUT_DIR / "accuracy_curve.png")
    plot_confusion_matrix(cm, OUT_DIR / "confusion_matrix.png")

    print("\n=== ÖZET ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Çıktılar: {OUT_DIR}")


if __name__ == "__main__":
    main()
