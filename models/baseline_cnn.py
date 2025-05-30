# src/models/baseline_cnn.py - Model architectures for systematic testing

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN for baseline experiments
    Purpose: Demonstrate underfitting with minimal architecture
    """
    def __init__(self, num_classes=7, dropout_rate=0.0):
        super(SimpleCNN, self).__init__()
        
        # Very simple architecture - likely to underfit
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate flattened size: 48x48 -> 24x24 -> 12x12
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))  # 48x48 -> 24x24
        x = self.pool(F.relu(self.conv2(x)))  # 24x24 -> 12x12
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_complexity_score(self):
        """Return a complexity score for analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

class DeepCNN(nn.Module):
    """
    Deeper CNN for advanced experiments
    Purpose: Show progression from simple to complex
    """
    def __init__(self, num_classes=7, dropout_rate=0.3, use_batch_norm=True):
        super(DeepCNN, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Progressive feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1: 48x48 -> 24x24
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Block 2: 24x24 -> 12x12
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Block 3: 12x12 -> 6x6
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Ensure consistent size
        x = self.adaptive_pool(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class OverfitCNN(nn.Module):
    """
    Deliberately overfitting model for analysis
    Purpose: Demonstrate overfitting scenario
    """
    def __init__(self, num_classes=7):
        super(OverfitCNN, self).__init__()
        
        # Very large model without regularization
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
        # Very large classifier - prone to overfitting
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 6 * 6, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNetBlock(nn.Module):
    """ResNet-style block for more advanced architectures"""
    def __init__(self, in_channels, out_channels, stride=1, use_batch_norm=True):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class CustomResNet(nn.Module):
    """
    Custom ResNet for FER task
    Purpose: Show advanced architecture techniques
    """
    def __init__(self, num_classes=7, dropout_rate=0.3, use_batch_norm=True):
        super(CustomResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(32, 32, 2, stride=1, use_batch_norm=use_batch_norm)
        self.layer2 = self._make_layer(32, 64, 2, stride=2, use_batch_norm=use_batch_norm)
        self.layer3 = self._make_layer(64, 128, 2, stride=2, use_batch_norm=use_batch_norm)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, use_batch_norm):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, use_batch_norm))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1, use_batch_norm))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class TransferLearningModel(nn.Module):
    """
    Transfer learning wrapper for pre-trained models
    Purpose: Demonstrate transfer learning techniques
    """
    def __init__(self, base_model_name='resnet18', num_classes=7, freeze_features=True):
        super(TransferLearningModel, self).__init__()
        
        if base_model_name == 'resnet18':
            import torchvision.models as models
            self.base_model = models.resnet18(pretrained=True)
            
            # Modify first layer for grayscale input
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Replace classifier
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_classes)
            
            # Freeze feature layers if requested
            if freeze_features:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                # Unfreeze classifier
                for param in self.base_model.fc.parameters():
                    param.requires_grad = True
                # Unfreeze first conv layer (we modified it)
                for param in self.base_model.conv1.parameters():
                    param.requires_grad = True
        
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
    
    def forward(self, x):
        return self.base_model(x)

class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models
    Purpose: Demonstrate ensemble techniques
    """
    def __init__(self, models_list, num_classes=7):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models_list)
        self.num_models = len(models_list)
        self.num_classes = num_classes
        
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(F.softmax(pred, dim=1))
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        # Convert back to logits for loss calculation
        ensemble_pred = torch.clamp(ensemble_pred, min=1e-8, max=1.0-1e-8)
        ensemble_logits = torch.log(ensemble_pred / (1 - ensemble_pred + 1e-8))
        
        return ensemble_logits

# Model factory function
def create_model(model_config):
    """
    Factory function to create models based on configuration
    """
    model_type = model_config.get('type', 'simple_cnn')
    num_classes = model_config.get('num_classes', 7)
    
    if model_type == 'simple_cnn':
        return SimpleCNN(
            num_classes=num_classes,
            dropout_rate=model_config.get('dropout_rate', 0.0)
        )
    
    elif model_type == 'deep_cnn':
        return DeepCNN(
            num_classes=num_classes,
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
    
    elif model_type == 'overfit_cnn':
        return OverfitCNN(num_classes=num_classes)
    
    elif model_type == 'custom_resnet':
        return CustomResNet(
            num_classes=num_classes,
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
    
    elif model_type == 'transfer_learning':
        return TransferLearningModel(
            base_model_name=model_config.get('base_model', 'resnet18'),
            num_classes=num_classes,
            freeze_features=model_config.get('freeze_features', True)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Analysis utilities
def analyze_model_complexity(model):
    """Analyze model complexity for documentation"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate approximate FLOPs (simplified)
    # This is a rough estimate for CNN models
    flops_estimate = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs: input_size * kernel_size^2 * in_channels * out_channels
            if hasattr(module, 'weight'):
                kernel_flops = module.weight.numel()
                # Assuming input size, this is approximate
                flops_estimate += kernel_flops * 48 * 48  # Rough estimate
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'estimated_flops': flops_estimate,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def compare_models(models_dict):
    """Compare multiple models side by side"""
    comparison = {}
    
    for name, model in models_dict.items():
        stats = analyze_model_complexity(model)
        comparison[name] = stats
    
    return comparison

# Example usage for systematic experiments
if __name__ == "__main__":
    # Create models for comparison
    models_to_compare = {
        'Simple CNN': SimpleCNN(),
        'Deep CNN': DeepCNN(),
        'Deep CNN (No BatchNorm)': DeepCNN(use_batch_norm=False),
        'Overfit CNN': OverfitCNN(),
        'Custom ResNet': CustomResNet(),
    }
    
    # Compare model complexities
    comparison = compare_models(models_to_compare)
    
    print("Model Complexity Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Parameters':<12} {'Trainable':<12} {'Size (MB)':<10}")
    print("-" * 80)
    
    for name, stats in comparison.items():
        print(f"{name:<20} {stats['total_parameters']:<12,} {stats['trainable_parameters']:<12,} {stats['model_size_mb']:<10.2f}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 48, 48)
    
    print("\nTesting forward pass...")
    for name, model in models_to_compare.items():
        try:
            output = model(dummy_input)
            print(f"{name}: Output shape {output.shape} ✓")
        except Exception as e:
            print(f"{name}: Error - {e}")