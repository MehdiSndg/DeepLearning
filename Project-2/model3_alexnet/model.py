"""
Model 3 — AlexNet (torchvision hazır mimari)

torchvision.models.alexnet kullanılır. ImageNet üzerinde önceden eğitilmiş
(pretrained=True) ağırlıklar yüklenir, son sınıflandırıcı katman CIFAR-10'un
10 sınıfı için yeniden tanımlanır (fine-tuning).

Not: AlexNet girişleri 3x224x224 bekler. CIFAR-10 görüntüleri (32x32)
train.py içinde transforms.Resize(224) ile büyütülür.
"""
import torch.nn as nn
from torchvision import models


def build_alexnet(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)

    # Son FC katmanını CIFAR-10 için değiştir (1000 -> 10)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
