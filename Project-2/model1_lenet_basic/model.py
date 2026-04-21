"""
Model 1 — LeNet-5 (temel sürüm)

CIFAR-10 (3 kanal, 32x32) için uyarlanmış klasik LeNet-5 mimarisi.
Sadece temel katmanlar: Conv2d, ReLU, MaxPool2d, Flatten, Linear.
BatchNorm veya Dropout YOK — bunlar Model 2'de gelecek.
"""
import torch.nn as nn


class LeNet5Basic(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Özellik çıkarıcı (konvolüsyon blokları)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),   # 32 -> 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 28 -> 14

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 14 -> 10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 10 -> 5
        )

        # Sınıflandırıcı (tam bağlantılı katmanlar)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
