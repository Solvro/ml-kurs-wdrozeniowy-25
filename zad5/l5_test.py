"""
Lista 5 - Zadanie 2: Transfer Learning na CIFAR-10
Wybór i analiza architektury ResNet18
"""

import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
# Ładujemy ResNet18 z najnowszymi wagami ImageNet (IMAGENET1K_V1)
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

print(f"Typ modelu: {type(model).__name__}")
print(model)

# =============================================================================
# 3. Analiza struktury modelu
# =============================================================================

print("\n" + "=" * 80)
print("ANALIZA STRUKTURY MODELU")
print("=" * 80)

# 3.1 Część konwolucyjna (backbone)
print("\n" + "-" * 40)
print("BACKBONE (część konwolucyjna):")
print("-" * 40)
print("""
Backbone składa się z:
1. conv1 - początkowa warstwa konwolucyjna (7x7, stride=2)
2. bn1 - Batch Normalization
3. relu - funkcja aktywacji
4. maxpool - Max Pooling (3x3, stride=2)
5. layer1 - pierwszy blok residualny (2 bloki BasicBlock, 64 filtry)
6. layer2 - drugi blok residualny (2 bloki BasicBlock, 128 filtrów)
7. layer3 - trzeci blok residualny (2 bloki BasicBlock, 256 filtrów)
8. layer4 - czwarty blok residualny (2 bloki BasicBlock, 512 filtrów)
9. avgpool - Adaptive Average Pooling
""")

# Wyświetlenie poszczególnych warstw backbone
print("Szczegóły warstw backbone:")
print(f"\nconv1: {model.conv1}")
print(f"bn1: {model.bn1}")
print(f"maxpool: {model.maxpool}")

# 3.2 Bloki residualne
print("\n" + "-" * 40)
print("STRUKTURA BLOKÓW RESIDUALNYCH:")
print("-" * 40)

for i, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4'], 1):
    layer = getattr(model, layer_name)
    print(f"\n{layer_name} (Blok {i}):")
    print(layer)

# 3.3 Ostatnia warstwa klasyfikacyjna
print("\n" + "-" * 40)
print("OSTATNIA WARSTWA KLASYFIKACYJNA (fc):")
print("-" * 40)
print(f"\n{model.fc}")
print(f"\nWejście: {model.fc.in_features} cech")
print(f"Wyjście: {model.fc.out_features} klas (ImageNet ma 1000 klas)")

# =============================================================================
# 4. Podsumowanie architektury
# =============================================================================

print("\n" + "=" * 80)
print("PODSUMOWANIE ARCHITEKTURY ResNet18")
print("=" * 80)

# Zliczanie parametrów
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"""
Model: ResNet18
Liczba parametrów: {total_params:,}
Parametry treowalne: {trainable_params:,}

STRUKTURA:
┌─────────────────────────────────────────────────────────────┐
│  INPUT (224x224x3)                                          │
├─────────────────────────────────────────────────────────────┤
│  conv1 (7x7, 64, stride=2) + BN + ReLU                      │
│  MaxPool (3x3, stride=2)                                    │
├─────────────────────────────────────────────────────────────┤
│  BACKBONE (część konwolucyjna):                             │
│  ├── layer1: 2x BasicBlock (64 filtry)                      │
│  ├── layer2: 2x BasicBlock (128 filtrów, downsampling)      │
│  ├── layer3: 2x BasicBlock (256 filtrów, downsampling)      │
│  └── layer4: 2x BasicBlock (512 filtrów, downsampling)      │
├─────────────────────────────────────────────────────────────┤
│  AdaptiveAvgPool2d (1x1)                                    │
├─────────────────────────────────────────────────────────────┤
│  GŁOWA KLASYFIKACYJNA:                                      │
│  └── fc: Linear(512 -> 1000)                                │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT (1000 klas ImageNet)                                │
└─────────────────────────────────────────────────────────────┘

BLOK RESIDUALNY (BasicBlock):
┌───────────────────┐
│   Input (x)       │──────────────────┐
├───────────────────┤                  │
│  Conv2d 3x3       │                  │
│  BatchNorm2d      │                  │
│  ReLU             │                  │ (skip connection)
│  Conv2d 3x3       │                  │
│  BatchNorm2d      │                  │
├───────────────────┤                  │
│    (+) ───────────│──────────────────┘
│  ReLU             │
├───────────────────┤
│   Output          │
└───────────────────┘
""")

# =============================================================================
# 5. Test przepuszczenia danych przez model
# =============================================================================

print("\n" + "=" * 80)
print("TEST PRZEPUSZCZENIA DANYCH")
print("=" * 80)

# Ustawiamy model w tryb ewaluacji
model.eval()

# Tworzymy przykładowe dane wejściowe (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print(f"\nWejście: {dummy_input.shape}")
print(f"Wyjście: {output.shape}")
print(f"Suma prawdopodobieństw (po softmax): {torch.softmax(output, dim=1).sum().item():.4f}")

print("\n" + "=" * 80)
print("ANALIZA ZAKOŃCZONA POMYŚLNIE!")
print("=" * 80)
