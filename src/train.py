import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import wandb
from tqdm import tqdm
import argparse
import sys

# Add the parent directory to path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.expression_cnn import get_model

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the image and convert to numpy array
        image = np.array(self.data.iloc[idx, 1].split(), dtype='uint8')
        image = image.reshape(48, 48)
        
        # Get the label
        label = int(self.data.iloc[idx, 0])
        
        # Convert to PIL Image for transformations
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_epoch(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [TRAIN]', leave=False)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        running_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Facial Expression Recognition Training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='directory containing the data')
    parser.add_argument('--model-dir', type=str, default='saved_models',
                        help='directory to save the model')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize WandB
    wandb.init(project="facial-expression-recognition", config=args)
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = get_model(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    # Data loading and transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets
    train_dataset = FER2013Dataset(
        csv_file=os.path.join(args.data_dir, 'train.csv'),
        transform=transform
    )
    
    val_dataset = FER2013Dataset(
        csv_file=os.path.join(args.data_dir, 'val.csv'),
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, device, train_loader, criterion, optimizer, epoch
        )
        
        # Evaluate on validation set
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics to WandB
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Save the best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, os.path.join(args.model_dir, 'best_model.pth'))
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    print(f'Training complete. Best validation accuracy: {best_val_accuracy:.2f}%')
    wandb.finish()

if __name__ == '__main__':
    main()
