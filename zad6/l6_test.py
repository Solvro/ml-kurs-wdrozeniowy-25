from torch.utils.data import DataLoader, random_split
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST), 
    transforms.PILToTensor()
])

dataset_trainval = datasets.OxfordIIITPet(
    root='./data',
    split='trainval',  
    target_types='segmentation',
    download=True,
    transform=transform,
    target_transform=target_transform
)

dataset_test = datasets.OxfordIIITPet(
    root='./data',
    split='test',  
    target_types='segmentation',
    download=True,
    transform=transform,
    target_transform=target_transform
)

train_size = int(0.8 * len(dataset_trainval))
val_size = len(dataset_trainval) - train_size
dataset_train, dataset_val = random_split(dataset_trainval, [train_size, val_size])


# DataLoadery
batch_size = 32

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

sample_img, sample_mask = dataset_trainval[0]


class DoubleConv(nn.Module):
    """Podwójna konwolucja używana w blokach sieci"""
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = None):
        super().__init__()

        if not hidden_channels:
            hidden_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module): # oddzielne sieci dla przejrzystości i łatwości dalszego użycia
    def __init__(self, in_channels: int = 3, bottleneck_channels: int = 1024):
        super().__init__()
        # Enkoder - zmniejszamy wymiary przestrzenne, zwiększamy liczbę kanałów
        self.enc1 = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.enc2 = nn.Sequential(
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.enc3 = nn.Sequential(
            DoubleConv(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.enc4 = nn.Sequential(
            DoubleConv(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, bottleneck_channels)
        self.bottleneck_channels = bottleneck_channels

    def forward(self, x):
        x1 = self.enc1(x)    # (B, 64, H/2, W/2)
        x2 = self.enc2(x1)   # (B, 128, H/4, W/4)
        x3 = self.enc3(x2)   # (B, 256, H/8, W/8)
        x4 = self.enc4(x3)   # (B, 512, H/16, W/16)
        b = self.bottleneck(x4)  # (B, 1024, H/16, W/16)
        return b


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, bottleneck_channels: int = 1024):
        super().__init__()
        # Dekoder - zwiększamy wymiary przestrzenne, zmniejszamy liczbę kanałów
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 512, kernel_size=2, stride=2),
            DoubleConv(512, 512)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(256, 256)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(128, 128)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(64, 64)
        )
        
        # Warstwa wyjściowa
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d4 = self.dec4(x)    # (B, 512, H/8, W/8)
        d3 = self.dec3(d4)   # (B, 256, H/4, W/4)
        d2 = self.dec2(d3)   # (B, 128, H/2, W/2)
        d1 = self.dec1(d2)   # (B, 64, H, W)
        output = self.output(d1)  # (B, 3, H, W)
        output = self.sigmoid(output)
        return output
  
  
class CNNAutoencoder(nn.Module): # łączymy tutaj logikę enkodera i dekodera
    def __init__(self, in_channels: int = 3, out_channels: int = 3, bottleneck_channels: int = 1024):
        super().__init__()
        self.encoder = Encoder(in_channels, bottleneck_channels)
        self.decoder = Decoder(out_channels, bottleneck_channels)
        self.bottleneck_channels = bottleneck_channels

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def embeddings(self, x): # użyj tylko enkodera i zwróc spłaszczony wynik
        encoded = self.encoder(x)
        # Spłaszczamy do wektora
        return encoded.view(encoded.size(0), -1)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, _ in dataloader:
        images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, images)
            
            running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def get_reconstructions(model, dataloader, device, num_samples=8):
    model.eval()
    images_list = []
    reconstructions_list = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            images_list.append(images.cpu())
            reconstructions_list.append(outputs.cpu())
            
            if len(images_list) * images.size(0) >= num_samples:
                break
    
    images_tensor = torch.cat(images_list, dim=0)[:num_samples]
    reconstructions_tensor = torch.cat(reconstructions_list, dim=0)[:num_samples]
    
    return images_tensor, reconstructions_tensor


def visualize_reconstructions(original, reconstructed, epoch, bottleneck_size):
    
    num_images = min(8, original.size(0))
    fig, axes = plt.subplots(2, num_images, figsize=(20, 5))
    
    for i in range(num_images):
        # Oryginalny obraz
        axes[0, i].imshow(original[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Oryginał', fontsize=12)
        
        # Zrekonstruowany obraz
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Rekonstrukcja', fontsize=12)
    
    plt.suptitle(f'Bottleneck={bottleneck_size}, Epoka {epoch}', fontsize=14)
    plt.tight_layout()
    plt.close()


def plot_training_history(train_losses, val_losses, bottleneck_size):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoka', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'Historia treningu - Bottleneck={bottleneck_size}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()


def train_autoencoder(bottleneck_size, num_epochs=20, learning_rate=0.001, save_reconstructions_every=5):

    print(f"")
    print(f"Trenuję autoencoder z bottleneck_size={bottleneck_size}")
    print(f"")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNNAutoencoder(in_channels=3, out_channels=3, bottleneck_channels=bottleneck_size)
    model = model.to(device)
        
    # Loss i optimizer
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Historie
    train_losses = []
    val_losses = []
    
    # Trening
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoka {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Zapisz rekonstrukcje co kilka epok
        if epoch % save_reconstructions_every == 0 or epoch == 1 or epoch == num_epochs:
            original, reconstructed = get_reconstructions(model, val_loader, device)
            visualize_reconstructions(original, reconstructed, epoch, bottleneck_size)
           
    
    # Wykres historii treningu
    plot_training_history(train_losses, val_losses, bottleneck_size)
    
    # Oblicz test loss
    test_loss = validate_epoch(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.6f}\n")
    
    return model, train_losses, val_losses, test_loss


if __name__ == '__main__':
    # Eksperymenty z różnymi rozmiarami bottlenecka
    bottleneck_sizes = [4, 16, 32]
    
    results = {}
    
    for bottleneck_size in bottleneck_sizes:
        model, train_losses, val_losses, test_loss = train_autoencoder(
            bottleneck_size=bottleneck_size,
            num_epochs=20,
            learning_rate=0.001,
            save_reconstructions_every=5
        )
        results[bottleneck_size] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_loss': test_loss
        }
    
    # Porównanie wszystkich eksperymentów
    print("")
    print("PODSUMOWANIE EKSPERYMENTÓW")
    print("")
    
    plt.figure(figsize=(12, 6))
    
    for bottleneck_size in bottleneck_sizes:
        val_losses = results[bottleneck_size]['val_losses']
        test_loss = results[bottleneck_size]['test_loss']
        plt.plot(val_losses, label=f'Bottleneck={bottleneck_size}', linewidth=2, marker='o', markersize=4)
        final_val_loss = val_losses[-1]
        print(f"Bottleneck={bottleneck_size:3d} - Końcowy Val Loss: {final_val_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    plt.xlabel('Epoka', fontsize=12)
    plt.ylabel('Validation Loss (MSE)', fontsize=12)
    plt.title('Porównanie różnych rozmiarów bottlenecka', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    
    






