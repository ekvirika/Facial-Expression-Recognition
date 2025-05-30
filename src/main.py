# main.py - Main training script with Wandb integration

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Custom imports (you'll create these files)
from src.data.dataset import FERDataset
from src.models.baseline_cnn import SimpleCNN, DeepCNN
from src.training.trainer import FERTrainer
from src.evaluation.metrics import calculate_metrics

class FERExperiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize Wandb
        wandb.init(
            project="fer-challenge",
            name=config['experiment_name'],
            config=config
        )
        
    def prepare_data(self):
        """Prepare train and validation datasets"""
        # Data transforms with progressive complexity
        if self.config['augmentation_level'] == 'none':
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        elif self.config['augmentation_level'] == 'light':
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:  # heavy
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load datasets
        train_dataset = FERDataset('train', transform=train_transform)
        val_dataset = FERDataset('val', transform=val_transform)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    def create_model(self):
        """Create model based on config"""
        if self.config['model_type'] == 'simple_cnn':
            model = SimpleCNN(
                num_classes=7,
                dropout_rate=self.config['dropout_rate']
            )
        elif self.config['model_type'] == 'deep_cnn':
            model = DeepCNN(
                num_classes=7,
                dropout_rate=self.config['dropout_rate'],
                use_batch_norm=self.config['use_batch_norm']
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Log model info to Wandb
        wandb.log({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params
        })
        
        return model
    
    def setup_training(self, model):
        """Setup optimizer, scheduler, and loss function"""
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        # Learning rate scheduler
        if self.config['use_scheduler']:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            scheduler = None
            
        return criterion, optimizer, scheduler
    
    def train_epoch(self, model, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, model, criterion):
        """Validate the model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def run_experiment(self):
        """Run the complete experiment"""
        print(f"Starting experiment: {self.config['experiment_name']}")
        
        # Prepare data and model
        self.prepare_data()
        model = self.create_model()
        criterion, optimizer, scheduler = self.setup_training(model)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate(model, criterion)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Calculate overfitting score
            overfitting_score = val_loss - train_loss
            
            # Log to Wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "overfitting_score": overfitting_score
            })
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Overfitting Score: {overfitting_score:.4f}")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_accuracy': val_acc
                }, f"results/models/{self.config['experiment_name']}_best.pth")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation and visualizations
        self.create_final_report(model, val_preds, val_targets, 
                               train_losses, val_losses, train_accs, val_accs)
        
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def create_final_report(self, model, val_preds, val_targets, 
                          train_losses, val_losses, train_accs, val_accs):
        """Create comprehensive final report"""
        
        # Class names for FER2013
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # 1. Confusion Matrix
        cm = confusion_matrix(val_targets, val_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Log to Wandb
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.savefig(f"results/plots/{self.config['experiment_name']}_confusion_matrix.png")
        plt.close()
        
        # 2. Training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        wandb.log({"training_curves": wandb.Image(fig)})
        plt.savefig(f"results/plots/{self.config['experiment_name']}_training_curves.png")
        plt.close()
        
        # 3. Classification Report
        report = classification_report(val_targets, val_preds, 
                                     target_names=class_names, output_dict=True)
        
        # Log per-class metrics
        for class_name in class_names:
            if class_name in report:
                wandb.log({
                    f"precision_{class_name}": report[class_name]['precision'],
                    f"recall_{class_name}": report[class_name]['recall'],
                    f"f1_{class_name}": report[class_name]['f1-score']
                })
        
        # Log overall metrics
        wandb.log({
            "macro_avg_precision": report['macro avg']['precision'],
            "macro_avg_recall": report['macro avg']['recall'],
            "macro_avg_f1": report['macro avg']['f1-score'],
            "weighted_avg_precision": report['weighted avg']['precision'],
            "weighted_avg_recall": report['weighted avg']['recall'],
            "weighted_avg_f1": report['weighted avg']['f1-score']
        })
        
        print("\nClassification Report:")
        print(classification_report(val_targets, val_preds, target_names=class_names))

# Example usage and configuration
if __name__ == "__main__":
    # Define different experiment configurations
    experiments = [
        {
            'experiment_name': 'baseline_simple_cnn_v1',
            'model_type': 'simple_cnn',
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10,
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'optimizer': 'adam',
            'weight_decay': 0.0,
            'use_scheduler': False,
            'augmentation_level': 'none'
        },
        {
            'experiment_name': 'baseline_simple_cnn_v2_dropout',
            'model_type': 'simple_cnn',
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10,
            'dropout_rate': 0.3,
            'use_batch_norm': False,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'use_scheduler': True,
            'augmentation_level': 'light'
        },
        {
            'experiment_name': 'deep_cnn_v1_overfit_demo',
            'model_type': 'deep_cnn',
            'batch_size': 32,
            'learning_rate': 0.01,
            'epochs': 100,
            'patience': 50,  # Allow overfitting
            'dropout_rate': 0.0,  # No regularization
            'use_batch_norm': False,
            'optimizer': 'sgd',
            'weight_decay': 0.0,
            'use_scheduler': False,
            'augmentation_level': 'none'
        }
    ]
    
    # Run experiments
    results = {}
    for config in experiments:
        experiment = FERExperiment(config)
        best_acc = experiment.run_experiment()
        results[config['experiment_name']] = best_acc
        wandb.finish()  # End current run
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    for exp_name, acc in results.items():
        print(f"{exp_name}: {acc:.2f}%")