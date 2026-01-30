import unittest
import torch
import torch.nn as nn
from model import FlexibleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class TestFlexibleCNN(unittest.TestCase):
    
    def test_basic_structure(self):
        """Test minimal requirement: 2 conv blocks and FC classifier"""
        model = FlexibleCNN() # Default has 2 blocks
        
        # Check for Conv2d layers
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        self.assertTrue(len(conv_layers) >= 2, "Should have at least 2 conv layers")
        
        # Check for Linear layer (Classifier)
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.assertTrue(len(linear_layers) >= 1, "Should have at least 1 linear layer")

    def test_pooling_options(self):
        """Test enabling/disabling pooling types"""
        # Test Max Pooling
        model_max = FlexibleCNN(pooling_type='max')
        max_pools = [m for m in model_max.modules() if isinstance(m, nn.MaxPool2d)]
        self.assertTrue(len(max_pools) > 0, "Max Pooling should be present")
        
        # Test Avg Pooling
        model_avg = FlexibleCNN(pooling_type='avg')
        avg_pools = [m for m in model_avg.modules() if isinstance(m, nn.AvgPool2d)]
        self.assertTrue(len(avg_pools) > 0, "Avg Pooling should be present")
        
        # Test No Pooling (check features specifically)
        model_none = FlexibleCNN(pooling_type=None)
        pools = [m for m in model_none.features.modules() if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d))]
        self.assertEqual(len(pools), 0, "No pooling should be present in features")

    def test_flatten_vs_gap(self):
        """Test switching between Flatten and Global Average Pooling"""
        # Case 1: Flatten
        model_flat = FlexibleCNN(use_global_avg_pooling=False, image_size=(3, 32, 32))
        dummy_input = torch.randn(1, 3, 32, 32)
        out_flat = model_flat(dummy_input)
        self.assertEqual(len(out_flat.shape), 2)
        # Check if GAP layer is NOT used
        self.assertFalse(model_flat.use_global_avg_pooling)
        
        # Case 2: GAP
        model_gap = FlexibleCNN(use_global_avg_pooling=True, image_size=(3, 32, 32))
        out_gap = model_gap(dummy_input)
        self.assertEqual(len(out_gap.shape), 2)
        self.assertTrue(model_gap.use_global_avg_pooling)

    def test_regularization(self):
        """Test BatchNorm and Dropout toggles"""
        model = FlexibleCNN(use_batch_norm=True, use_dropout=True)
        
        bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        drops = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]
        
        self.assertTrue(len(bns) > 0, "BatchNorm should be enabled")
        self.assertTrue(len(drops) > 0, "Dropout should be enabled")
        
        model_off = FlexibleCNN(use_batch_norm=False, use_dropout=False)
        bns_off = [m for m in model_off.modules() if isinstance(m, nn.BatchNorm2d)]
        drops_off = [m for m in model_off.modules() if isinstance(m, nn.Dropout2d)]
        
        self.assertEqual(len(bns_off), 0, "BatchNorm should be disabled")
        self.assertEqual(len(drops_off), 0, "Dropout should be disabled")

    def test_architecture_config(self):
        """Test changing number of layers and conv params"""
        config = [
            {'out_channels': 16, 'kernel_size': 5, 'stride': 2, 'padding': 2},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0}
        ]
        model = FlexibleCNN(conv_config=config, image_size=(3, 32, 32))
        
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        self.assertEqual(len(convs), 3, "Should have 3 conv layers")
        
        self.assertEqual(convs[0].out_channels, 16)
        self.assertEqual(convs[0].stride, (2, 2))
        self.assertEqual(convs[2].padding, (0, 0))

def train_dummy():
    print("\nRunning dummy training loop to verify training capability...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # data/ is inside zad4/ based on workspace info from user context?
    # Context says:
    # zad4/
    #    data/cifar-10-batches-py/
    
    # But current working dir is C:\Users\RODO\Desktop\PROJEKTY\ml-wdrożenie
    # So path to data for zad4 scripts should be relative to CWD if we run from CWD.
    # Note: zad4/data/cifar-10-batches-py exists.
    # datasets.CIFAR10 defaults to looking for 'cifar-10-batches-py' folder inside the root provided.
    # If root='./zad4/data', it looks for './zad4/data/cifar-10-batches-py'.
    
    data_path = "./zad4/data"
    
    try:
        # Check if data exists, otherwise fallback to download=True but be careful.
        # Given context, it's likely downloaded.
        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
        
        model = FlexibleCNN(use_batch_norm=True, pooling_type='max')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Train for 1 batch just to verify it runs
        data_iter = iter(trainloader)
        images, labels = next(data_iter)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print("Training step successful. Loss:", loss.item())
        
    except Exception as e:
        print(f"Training check failed: {e}")

if __name__ == '__main__':
    # Run tests
    print("Running Unit Tests for FlexibleCNN...")
    unittest.main(exit=False)
    
    # Run simple training check
    train_dummy()



