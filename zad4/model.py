import torch
import torch.nn as nn

class FlexibleCNN(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=10, 
                 conv_config=None, # List of dicts, e.g., [{'out_channels': 32, 'kernel_size': 3}, ...]
                 pooling_type=None, # 'max' or 'avg' or None
                 pooling_every_n_layers=1, # Add pooling after every N layers
                 use_global_avg_pooling=False,
                 use_batch_norm=False,
                 use_dropout=False,
                 dropout_prob=0.5,
                 image_size=(3, 32, 32) # C, H, W
                 ):
        super().__init__()
        
        # Default config: 2 layers if none provided
        if conv_config is None:
            conv_config = [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ]
            
        self.features = nn.Sequential()
        in_c = input_channels
        
        for i, layer_conf in enumerate(conv_config):
            out_c = layer_conf.get('out_channels', 32 * (i + 1))
            k = layer_conf.get('kernel_size', 3)
            s = layer_conf.get('stride', 1)
            p = layer_conf.get('padding', 1)
            d = layer_conf.get('dilation', 1)
            
            # Add Conv Layer
            self.features.add_module(f'conv_{i}', nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d))
            
            # Add Batch Norm
            if use_batch_norm:
                self.features.add_module(f'bn_{i}', nn.BatchNorm2d(out_c))
                
            # Add Activation
            self.features.add_module(f'relu_{i}', nn.ReLU())
            
            # Add Dropout
            if use_dropout:
                self.features.add_module(f'dropout_{i}', nn.Dropout2d(p=dropout_prob))
                
            # Add Pooling
            # Only add if pooling_type is specified AND (we are at the right interval OR it's specified per layer)
            # Simplification: Add pooling if requested after layer (could require more config for exact placement)
            if pooling_type and (i + 1) % pooling_every_n_layers == 0:
                if pooling_type == 'max':
                    self.features.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=2, stride=2))
                elif pooling_type == 'avg':
                    self.features.add_module(f'pool_{i}', nn.AvgPool2d(kernel_size=2, stride=2))
            
            in_c = out_c

        self.use_global_avg_pooling = use_global_avg_pooling
        
        # Calculate classifier input features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_size)
            dummy_out = self.features(dummy_input)
            
        if use_global_avg_pooling:
            self.classifier_input = in_c # After GAP, it is C x 1 x 1 -> C
        else:
            self.classifier_input = dummy_out.view(1, -1).shape[1]
            
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        
        if self.use_global_avg_pooling:
            # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        else:
            # Flatten: (B, C, H, W) -> (B, C*H*W)
            x = torch.flatten(x, 1)
            
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
import torch.nn.functional as F
