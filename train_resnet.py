"""
ResNet Training Script for Facial Expression Recognition
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import wandb
from tqdm import tqdm

# Import your custom ResNet model
from models.resnet_custom import resnet18_fer

# Configuration
class Config:
    # Model
    model_name = 'resnet18'
    num_classes = 7
    
    # Training
    batch_size = 64
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-4
    dropout_rate = 0.5
    
    # Data
    img_size = 48
    
    # Augmentation
    use_augmentation = True
    
    # Early Stopping
    patience = 5
    min_delta = 0.001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Weights & Biases
def init_wandb():
    wandb.init(
        project="facial-expression-recognition",
        entity="your-wandb-username",
        config={
            "architecture": "ResNet18",
            "dataset": "FER2013",
            "epochs": Config.epochs,
            "batch_size": Config.batch_size,
            "learning_rate": Config.lr,
        }
    )

# Data Augmentation
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transforms, val_transforms

# Training Loop
def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize with mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Validation Loop
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def main():
    # Initialize config
    config = Config()
    
    # Initialize Weights & Biases
    init_wandb()
    
    # Data loading and transformations
    train_transforms, val_transforms = get_transforms()
    
    # Load your dataset here (replace with your dataset loading code)
    # train_dataset = YourDataset(..., transform=train_transforms)
    # val_dataset = YourDataset(..., transform=val_transforms)
    
    # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
    #                         shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
    #                       shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = resnet18_fer(num_classes=config.num_classes)
    model = model.to(config.device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-" * 10)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                          optimizer, scaler, config.device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc + config.min_delta:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    main()
