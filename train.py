import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Import from our other .py files
import config
from model import CNN_LSTM
from preprocess import get_dataloaders

def calculate_class_weights(y_train):
    """Calculates class weights for imbalanced dataset."""
    print("Calculating class weights...")
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(config.DEVICE)
    
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
    print("Class weights for loss function:")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i} ({class_names[i]}): {weight:.4f}")
        
    return class_weights_tensor

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    """Saves plots for training loss and accuracy."""
    print(f"Saving training plots to {config.TRAIN_PLOTS_PATH}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.TRAIN_PLOTS_PATH)
    print("Training plots saved.")

def main():
    print(f"Using device: {config.DEVICE}")
    
    # 1. Get DataLoaders and y_train (for weights)
    train_loader, test_loader, y_train = get_dataloaders(
        config.TRAIN_FILE, 
        config.TEST_FILE, 
        config.BATCH_SIZE
    )
    
    # 2. Initialize Model
    model = CNN_LSTM().to(config.DEVICE)
    
    # 3. Define Loss Function (with class weights) and Optimizer
    class_weights = calculate_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 4. Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.LR_FACTOR, 
        patience=config.LR_PATIENCE
    )
    
    # 5. Training Loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    print("\nStarting training...\n")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("=" * 60)
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, test_loader, criterion, config.DEVICE)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f'{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {val_loss:<12.4f} {val_acc:<12.2f}% ⭐ BEST')
        else:
            print(f'{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {val_loss:<12.4f} {val_acc:<12.2f}%')
            
    print("\n" + "=" * 60)
    print(f'Training complete! Best validation accuracy: {best_val_acc:.2f}%')
    print(f"Best model saved to {config.MODEL_PATH}")
    print("=" * 60)
    
    # 6. Plot and save training results
    plot_training_results(train_losses, val_losses, train_accs, val_accs)

if __name__ == '__main__':
    main()
