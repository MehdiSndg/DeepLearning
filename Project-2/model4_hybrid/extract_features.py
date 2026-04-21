"""
Model 4 — Özellik çıkarımı (AlexNet -> .npy)

Eğitilmiş Model 3 checkpoint'i yüklenir, son sınıflandırıcı katman kaldırılır
(classifier[:-1] korunur -> 4096 boyutlu vektör çıkar). Tüm CIFAR-10 train
ve test örnekleri bu ağdan geçirilerek özellik matrisleri oluşturulur.

Çıktılar (features/):
  - X_train.npy   (50000, 4096)
  - y_train.npy   (50000,)
  - X_test.npy    (10000, 4096)
  - y_test.npy    (10000,)

Boyut ve uzunluk bilgileri outputs/feature_info.txt'e de yazılır (PDF şartı).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_PATH = PROJECT_DIR / "model3_alexnet" / "checkpoint.pth"
FEAT_DIR = SCRIPT_DIR / "features"
OUT_DIR = SCRIPT_DIR / "outputs"
FEAT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Model 3'teki build_alexnet fonksiyonunu import etmek için path ekle
sys.path.insert(0, str(PROJECT_DIR / "model3_alexnet"))
from model import build_alexnet  # noqa: E402

BATCH_SIZE = 128
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


def load_feature_extractor() -> nn.Module:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {CKPT_PATH}\n"
            f"Önce model3_alexnet/train.py çalıştırılmalı."
        )
    model = build_alexnet(num_classes=10, pretrained=False)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    # Son Linear katmanı kaldır -> 4096 boyutlu vektör çıkışı
    # AlexNet.classifier: [Dropout, Linear(9216,4096), ReLU, Dropout, Linear(4096,4096), ReLU, Linear(4096,10)]
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def extract(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        f = model(x)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    print(f"Cihaz: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")

    model = load_feature_extractor()
    train_loader, test_loader = get_loaders()

    print("Train özellikleri çıkarılıyor...")
    t0 = time.time()
    X_train, y_train = extract(model, train_loader)
    print(f"  süre: {time.time() - t0:.1f} sn")

    print("Test özellikleri çıkarılıyor...")
    t0 = time.time()
    X_test, y_test = extract(model, test_loader)
    print(f"  süre: {time.time() - t0:.1f} sn")

    np.save(FEAT_DIR / "X_train.npy", X_train)
    np.save(FEAT_DIR / "y_train.npy", y_train)
    np.save(FEAT_DIR / "X_test.npy", X_test)
    np.save(FEAT_DIR / "y_test.npy", y_test)

    # Boyut/uzunluk bilgisi (PDF şartı: "boyutu ve uzunluğu açıkça projede yazdırılmalıdır")
    info_lines = [
        "Model 4 — Özellik Seti Bilgileri",
        "=" * 50,
        f"Kaynak ağ: AlexNet (model3_alexnet/checkpoint.pth)",
        f"Çıkış boyutu (her örnek): {X_train.shape[1]} boyutlu vektör",
        "",
        f"X_train.npy: shape={X_train.shape}, dtype={X_train.dtype}, length={len(X_train)}",
        f"y_train.npy: shape={y_train.shape}, dtype={y_train.dtype}, length={len(y_train)}",
        f"X_test.npy : shape={X_test.shape}, dtype={X_test.dtype}, length={len(X_test)}",
        f"y_test.npy : shape={y_test.shape}, dtype={y_test.dtype}, length={len(y_test)}",
        "",
        f"X_train bellek boyutu: {X_train.nbytes / 1024 / 1024:.2f} MB",
        f"X_test  bellek boyutu: {X_test.nbytes / 1024 / 1024:.2f} MB",
    ]
    info_text = "\n".join(info_lines)
    print("\n" + info_text)
    (OUT_DIR / "feature_info.txt").write_text(info_text)
    print(f"\nÖzellik dosyaları: {FEAT_DIR}")


if __name__ == "__main__":
    main()
