from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


transform = transforms.Compose([
    transforms.ToTensor()
])

# Pełny training set
trainset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Podział na train i validation (80/20)
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_set, valset = random_split(trainset, [train_size, val_size])

# Test set
testset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# DataLoaders
batch_size = 64
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=use_cuda
)

val_loader = DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=use_cuda
)

test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=use_cuda
)

class ParameterizedMLP(torch.nn.Module):
    def __init__(self, hidden_layers, use_batch_norm=False, dropout_rate=0.0, input_size=28*28, output_size=10, activation_fn=torch.nn.ReLU()):
        super().__init__()
        
        layers = []
        current_input_size = input_size
        
        for i, layer_size in enumerate(hidden_layers):
            # Warstwa liniowa
            layers.append(torch.nn.Linear(current_input_size, layer_size))
            
            # Batch Normalization
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(layer_size))
            
            # Funkcja aktywacji
            layers.append(activation_fn)

            # Dropout
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(p=dropout_rate))
            
            current_input_size = layer_size
            
        # Warstwa wyjściowa
        layers.append(torch.nn.Linear(current_input_size, output_size))
        
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        # Spłaszczenie obrazu z (batch_size, 1, 28, 28) do (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.model(x)


# --- Przykłady użycia ---

# Przykład 1: Prosty model z dwiema warstwami ukrytymi (128 i 64 neurony), bez Batch Normalization
model_1 = ParameterizedMLP(hidden_layers=[128, 64], use_batch_norm=False)
print("Architektura Modelu 1:", model_1)

# --- Kod treningowy ---

import random
import numpy as np

def set_seed(seed):
    """Ustawia ziarno losowości dla zapewnienia powtarzalności wyników."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, test_loader, epochs, optimizer, loss_fn, patience, model_save_path, scheduler=None, l1_lambda=0.0, l2_lambda=0.0):
    """
    Funkcja do trenowania i ewaluacji modelu z early stoppingiem.

    :param model: Model PyTorch do trenowania.
    :param train_loader: DataLoader dla danych treningowych.
    :param val_loader: DataLoader dla danych walidacyjnych.
    :param test_loader: DataLoader dla danych testowych.
    :param epochs: Maksymalna liczba epok.
    :param optimizer: Optymalizator.
    :param loss_fn: Funkcja straty.
    :param patience: Liczba epok bez poprawy na zbiorze walidacyjnym, po której trening zostanie przerwany.
    :param model_save_path: Ścieżka do zapisu najlepszego modelu.
    :param scheduler: Opcjonalny scheduler zmiany learning rate.
    :param l1_lambda: Współczynnik regularyzacji L1.
    :param l2_lambda: Współczynnik regularyzacji L2.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # --- Trening ---
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Regularyzacja L1
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

            # Regularyzacja L2
            if l2_lambda > 0:
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss += l2_lambda * l2_norm

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        
        if scheduler:
            scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)

        # --- Walidacja ---
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)

        print(f"Epoka {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # --- Early Stopping i zapis modelu ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model zapisany. Najlepsza strata walidacyjna: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping! Brak poprawy przez {patience} epok.")
                break
    
    # --- Ewaluacja na zbiorze testowym ---
    print("\nTrening zakończony. Ładowanie najlepszego modelu i ewaluacja na zbiorze testowym...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / len(test_loader.dataset)
    print("-" * 30)
    print(f"Wyniki na zbiorze testowym:")
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
    print("-" * 30)


# --- Użycie kodu treningowego ---
if __name__ == '__main__':
    # Ustawienie ziarna losowości
    set_seed(42)

    # Parametry
    EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 5
    MODEL_SAVE_PATH = "best_model.pth"
    USE_SCHEDULER = True # Flaga do włączania schedulera
    L1_LAMBDA = 0.0
    L2_LAMBDA = 0.001 # Przykładowa wartość dla L2
    ACTIVATION_FUNCTION = torch.nn.ReLU() # Możesz zmienić na np. torch.nn.Tanh() lub torch.nn.Sigmoid()

    # Inicjalizacja modelu
    # Możesz tutaj eksperymentować z architekturą
    model_to_train = ParameterizedMLP(
        hidden_layers=[128, 64], 
        use_batch_norm=True, 
        dropout_rate=0.3,
        activation_fn=ACTIVATION_FUNCTION
    )

    # Optymalizator i funkcja straty
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Scheduler (opcjonalnie)
    scheduler = None
    if USE_SCHEDULER:
        # Cosine Annealing LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        print("Użycie CosineAnnealingLR włączone.")

    # Uruchomienie treningu
    train_model(
        model=model_to_train,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        optimizer=optimizer,
        loss_fn=loss_fn,
        patience=PATIENCE,
        model_save_path=MODEL_SAVE_PATH,
        scheduler=scheduler,
        l1_lambda=L1_LAMBDA,
        l2_lambda=L2_LAMBDA
    )