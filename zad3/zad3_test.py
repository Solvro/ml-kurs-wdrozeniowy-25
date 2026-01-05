from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Definiujemy transformacje
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.FashionMNIST(
    root="./zad3/data",
    train=True,
    download=True,
    transform=transform
)

testset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print(f"Liczba obrazów treningowych: {len(trainset)}")
print(f"Liczba obrazów testowych: {len(testset)}")

# Nazwy klas w FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Funkcja do wyświetlania obrazów
def show_images(dataset, n_images=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n_images):
        img, label = dataset[i]
        
        # Konwersja z tensora do numpy i denormalizacja
        img = img.numpy().squeeze()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{class_names[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Wyświetlamy przykładowe obrazy z zestawu treningowego
print("\nPrzykładowe obrazy z zestawu treningowego:")
show_images(trainset, n_images=10)