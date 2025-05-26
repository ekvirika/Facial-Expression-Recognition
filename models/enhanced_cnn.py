import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        return out

class EnhancedExpressionCNN(nn.Module):
    """
    Enhanced CNN model with residual connections for facial expression recognition.
    """
    def __init__(self, num_classes=7):
        super(EnhancedExpressionCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, 48, 48)
        x = F.relu(self.bn1(self.conv1(x)))  # -> (batch_size, 64, 24, 24)
        x = self.maxpool(x)  # -> (batch_size, 64, 12, 12)
        
        x = self.layer1(x)  # -> (batch_size, 64, 12, 12)
        x = self.layer2(x)  # -> (batch_size, 128, 6, 6)
        x = self.layer3(x)  # -> (batch_size, 256, 3, 3)
        x = self.layer4(x)  # -> (batch_size, 512, 2, 2)
        
        x = self.avgpool(x)  # -> (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # -> (batch_size, 512)
        x = self.fc(x)  # -> (batch_size, num_classes)
        
        return x

def get_enhanced_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Helper function to create and move model to the specified device
    """
    model = EnhancedExpressionCNN(num_classes=7)
    return model.to(device)

def get_optimizer(model, lr=0.001, weight_decay=1e-4):
    """
    Create optimizer with different learning rates for different layers
    """
    # Parameters of newly constructed modules have requires_grad=True by default
    param_groups = [
        {'params': model.conv1.parameters(), 'lr': lr * 0.1},
        {'params': model.bn1.parameters(), 'lr': lr * 0.1},
        {'params': model.layer1.parameters(), 'lr': lr * 0.5},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters()},
        {'params': model.fc.parameters()},
    ]
    
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, T_max=100, eta_min=1e-6):
    """
    Create learning rate scheduler
    """
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

if __name__ == "__main__":
    # Test the model with a sample input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a sample input tensor (batch_size=1, channels=1, height=48, width=48)
    x = torch.randn(1, 1, 48, 48).to(device)
    
    # Initialize model
    model = get_enhanced_model(device)
    print(model)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test optimizer
    optimizer = get_optimizer(model)
    print("\nOptimizer groups:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: lr={group['lr']}, params={len(group['params'])}")
    
    # Test scheduler
    scheduler = get_scheduler(optimizer)
    print("\nScheduler:", scheduler)
