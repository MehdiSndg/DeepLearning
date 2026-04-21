"""
Model 2 — LeNet-5 (iyileştirilmiş sürüm)

Model 1 ile TAM OLARAK AYNI hiperparametreleri (filtre sayıları, çekirdek
boyutları, FC birim sayıları) korur. Tek fark: her Conv ve FC katmanından
sonra BatchNorm ve FC katmanlarında Dropout eklenmiştir. PDF şartı gereği
bu iki model aynı train/test setinde karşılaştırılacaktır.

Ek katmanlar:
  - BatchNorm2d (Conv'lardan sonra)  -> iç kovaryans kaymasını azaltır
  - BatchNorm1d (FC'lerden sonra)    -> FC aktivasyonlarını stabilize eder
  - Dropout(p=0.5)                    -> overfitting'i azaltır
"""
import torch.nn as nn


class LeNet5Improved(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),    # 32 -> 28
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28 -> 14

            nn.Conv2d(6, 16, kernel_size=5),   # 14 -> 10
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 10 -> 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
