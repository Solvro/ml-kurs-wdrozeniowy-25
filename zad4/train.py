import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FlexibleCNN
import os

def train():
    # Wymuś użycie CPU dla powtarzalności lub jeśli brak GPU, chociaż kod obsługuje oba.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hiperparametry
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    # Przygotowanie danych (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Ścieżka do danych
    data_path = './zad4/data'
    
    print(f"Loading data from {data_path}...")
    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Konfiguracja modelu zgodnie z wymaganiami:
    # 1. Co najmniej 2 bloki (Conv + Activation)
    # 2. Sieć w pełni połączona jako klasyfikator
    # 3. Użycie Flatten (use_global_avg_pooling=False)
    
    # Przykładowa konfiguracja 2 warstwowa
    simple_config = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]

    model = FlexibleCNN(
        input_channels=3,
        num_classes=10, # CIFAR-10 ma 10 klas
        conv_config=simple_config,
        pooling_type='max', # Opcjonalnie dodajemy pooling dla lepszej zbieżności, choć nie jest to surowo wymagane w "min 2 bloki"
        use_global_avg_pooling=False, # Kluczowe wymaganie: Flatten zamiast GAP
        use_batch_norm=False, # Na razie prosta sieć
        use_dropout=False
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created. Total parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

        # Ewaluacja na zbiorze testowym
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] finished. Test Accuracy: {acc:.2f} %')

    print("Training finished.")
    save_path = './zad4/simple_cnn_cifar10.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()
