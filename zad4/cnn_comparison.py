import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import FlexibleCNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    train_data = []
    train_labels = []
    
    # Load separate batches
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_dict = unpickle(batch_file)
        train_data.append(batch_dict[b'data'])
        train_labels += batch_dict[b'labels']
        
    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)
    
    test_file = os.path.join(data_dir, 'test_batch')
    test_dict = unpickle(test_file)
    test_data = test_dict[b'data']
    test_labels = np.array(test_dict[b'labels'])
    
    # Reshape and normalize
    # CIFAR-10 is (N, 3072) -> (N, 3, 32, 32)
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    
    # Convert to implementation tensors
    train_x = torch.from_numpy(train_data)
    train_y = torch.LongTensor(train_labels)
    test_x = torch.from_numpy(test_data)
    test_y = torch.LongTensor(test_labels)
    
    return train_x, train_y, test_x, test_y

def create_dataloaders(batch_size=64):
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'cifar-10-batches-py')
    if not os.path.exists(data_dir):
        # Fallback if running from root
        data_dir = os.path.join('zad4', 'data', 'cifar-10-batches-py')
        
    print(f"Loading data from {data_dir}")
    train_x, train_y, test_x, test_y = load_cifar10_data(data_dir)
    
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_and_evaluate(model_config, train_loader, test_loader, epochs=10, lr=0.001):
    # Filter config key 'name' which is for logging but not for model init
    init_config = {k: v for k, v in model_config.items() if k != 'name'}
    model = FlexibleCNN(**init_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"Training model with config: {model_config.get('name', 'Custom')}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
    return history

def plot_comparison(results, title, metric='val_acc'):
    plt.figure(figsize=(10, 6))
    for label, history in results.items():
        plt.plot(history[metric], label=label)
    
    plt.title(f'{title} - {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}_{metric}.png")
    # plt.show()

if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders()
    
    # Define baseline config
    baseline_config = {
        'input_channels': 3,
        'num_classes': 10,
        'pooling_type': 'max',
        'use_global_avg_pooling': False,
        'use_batch_norm': False,
        'use_dropout': False,
        'conv_config': [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
    }
    
    # 1. Compare Average vs Max Pooling
    print("\n--- Experiment 1: Avg vs Max Pooling ---")
    results_exp1 = {}
    
    # Max Pooling (Baseline)
    cfg_max = baseline_config.copy()
    cfg_max['name'] = 'Max Pooling'
    results_exp1['Max Pooling'] = train_and_evaluate(cfg_max, train_loader, test_loader)
    
    # Avg Pooling
    cfg_avg = baseline_config.copy()
    cfg_avg['pooling_type'] = 'avg'
    cfg_avg['name'] = 'Avg Pooling'
    results_exp1['Avg Pooling'] = train_and_evaluate(cfg_avg, train_loader, test_loader)
    
    plot_comparison(results_exp1, 'Pooling Comparison')
    
    # 2. Global Avg Pooling vs Flatten
    print("\n--- Experiment 2: GAP vs Flatten ---")
    results_exp2 = {}
    
    # Flatten (Baseline)
    cfg_flat = baseline_config.copy()
    cfg_flat['name'] = 'Flatten'
    results_exp2['Flatten'] = train_and_evaluate(cfg_flat, train_loader, test_loader)
    
    # GAP
    cfg_gap = baseline_config.copy()
    cfg_gap['use_global_avg_pooling'] = True
    cfg_gap['name'] = 'Global Avg Pooling'
    # Increase layers/filters slightly for GAP to be effective usually
    # But sticking to direct comparison
    results_exp2['Global Avg Pooling'] = train_and_evaluate(cfg_gap, train_loader, test_loader)
    
    plot_comparison(results_exp2, 'GAP vs Flatten Comparison')
    
    # 3. Batch Normalization and Dropout
    print("\n--- Experiment 3: BN and Dropout ---")
    results_exp3 = {}
    
    # Baseline (None)
    results_exp3['None'] = results_exp1['Max Pooling'] # Reuse
    
    # BN only
    cfg_bn = baseline_config.copy()
    cfg_bn['use_batch_norm'] = True
    cfg_bn['name'] = 'Batch Norm'
    results_exp3['Batch Norm'] = train_and_evaluate(cfg_bn, train_loader, test_loader)
    
    # Dropout only
    cfg_drop = baseline_config.copy()
    cfg_drop['use_dropout'] = True
    cfg_drop['dropout_prob'] = 0.2
    cfg_drop['name'] = 'Dropout'
    results_exp3['Dropout'] = train_and_evaluate(cfg_drop, train_loader, test_loader)
    
    # Both
    cfg_both = baseline_config.copy()
    cfg_both['use_batch_norm'] = True
    cfg_both['use_dropout'] = True
    cfg_both['dropout_prob'] = 0.2
    cfg_both['name'] = 'BN + Dropout'
    results_exp3['BN + Dropout'] = train_and_evaluate(cfg_both, train_loader, test_loader)
    
    plot_comparison(results_exp3, 'Regularization Comparison')
    
    # 4. Layers and Filters
    print("\n--- Experiment 4: Layers and Filters ---")
    results_exp4 = {}
    
    # Baseline (2 layers: 32, 64)
    results_exp4['2 Layers (32, 64)'] = results_exp1['Max Pooling']
    
    # More filters
    cfg_more_filters = baseline_config.copy()
    cfg_more_filters['conv_config'] = [
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    cfg_more_filters['name'] = '2 Layers (64, 128)'
    results_exp4['2 Layers (64, 128)'] = train_and_evaluate(cfg_more_filters, train_loader, test_loader)
    
    # More layers
    cfg_more_layers = baseline_config.copy()
    cfg_more_layers['conv_config'] = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    cfg_more_layers['name'] = '3 Layers (32, 64, 128)'
    results_exp4['3 Layers (32, 64, 128)'] = train_and_evaluate(cfg_more_layers, train_loader, test_loader)
    
    # More layers - 4 Layers
    cfg_4_layers = baseline_config.copy()
    cfg_4_layers['conv_config'] = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    cfg_4_layers['name'] = '4 Layers (32, 64, 128, 256)'
    results_exp4['4 Layers (32, 64, 128, 256)'] = train_and_evaluate(cfg_4_layers, train_loader, test_loader)

    plot_comparison(results_exp4, 'Architecture Comparison')
    
    # 5. Stride, Padding, Dilation - Extended
    print("\n--- Experiment 5: Stride, Padding, Dilation (Extended) ---")
    results_exp5 = {}
    
    # --- Kernels ---
    # Kernel 3 (Baseline)
    results_exp5['Kernel 3 (Baseline)'] = results_exp1['Max Pooling']

    # Kernel 5
    cfg_k5 = baseline_config.copy()
    cfg_k5['conv_config'] = [
        {'out_channels': 32, 'kernel_size': 5, 'stride': 1, 'padding': 2},
        {'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'padding': 2}
    ]
    cfg_k5['name'] = 'Kernel 5'
    results_exp5['Kernel 5'] = train_and_evaluate(cfg_k5, train_loader, test_loader)
    
    # Kernel 7
    # Padding: (K-1)/2 = 3
    cfg_k7 = baseline_config.copy()
    cfg_k7['conv_config'] = [
        {'out_channels': 32, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'out_channels': 64, 'kernel_size': 7, 'stride': 1, 'padding': 3}
    ]
    cfg_k7['name'] = 'Kernel 7'
    results_exp5['Kernel 7'] = train_and_evaluate(cfg_k7, train_loader, test_loader)
    
    plot_comparison(results_exp5, 'Kernel Size Comparison')
    
    # --- Strides ---
    results_exp6 = {}
    
    # Stride 1 (Baseline)
    results_exp6['Stride 1 (Baseline)'] = results_exp1['Max Pooling']
    
    # Stride 2
    cfg_s2 = baseline_config.copy()
    cfg_s2['pooling_type'] = None
    cfg_s2['conv_config'] = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ]
    cfg_s2['name'] = 'Stride 2'
    results_exp6['Stride 2'] = train_and_evaluate(cfg_s2, train_loader, test_loader)

    plot_comparison(results_exp6, 'Stride Comparison')
    
    # --- Dilation ---
    results_exp7 = {}
    
    # Dilation 1 (Baseline)
    results_exp7['Dilation 1 (Baseline)'] = results_exp1['Max Pooling']
    
    # Dilation 2
    cfg_d2 = baseline_config.copy()
    cfg_d2['conv_config'] = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 2, 'dilation': 2},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 2, 'dilation': 2}
    ]
    cfg_d2['name'] = 'Dilation 2'
    results_exp7['Dilation 2'] = train_and_evaluate(cfg_d2, train_loader, test_loader)
    
    # Dilation 3
    # Padding needed: D*(K-1)/2 = 3*2/2 = 3
    cfg_d3 = baseline_config.copy()
    cfg_d3['conv_config'] = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 3, 'dilation': 3},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 3, 'dilation': 3}
    ]
    cfg_d3['name'] = 'Dilation 3'
    results_exp7['Dilation 3'] = train_and_evaluate(cfg_d3, train_loader, test_loader)
    
    plot_comparison(results_exp7, 'Dilation Comparison')
