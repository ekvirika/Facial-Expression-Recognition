import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpressionCNN(nn.Module):
    """
    A simple CNN model for facial expression recognition.
    Input: 1x48x48 grayscale image
    Output: 7 emotion classes
    """
    def __init__(self, num_classes=7):
        super(ExpressionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 48, 48)
        
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (batch_size, 32, 24, 24)
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (batch_size, 64, 12, 12)
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (batch_size, 128, 6, 6)
        
        # Fourth conv block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # -> (batch_size, 256, 3, 3)
        
        # Flatten the output
        x = x.view(-1, 256 * 3 * 3)  # -> (batch_size, 2304)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))  # -> (batch_size, 1024)
        x = self.dropout(F.relu(self.fc2(x)))  # -> (batch_size, 512)
        x = self.fc3(x)  # -> (batch_size, num_classes)
        
        return x

def get_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Helper function to create and move model to the specified device
    """
    model = ExpressionCNN(num_classes=7)
    return model.to(device)

if __name__ == "__main__":
    # Test the model with a sample input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a sample input tensor (batch_size=1, channels=1, height=48, width=48)
    x = torch.randn(1, 1, 48, 48).to(device)
    
    # Initialize model
    model = get_model(device)
    print(model)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
