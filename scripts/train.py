import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from colorama import Fore, Style, init
from shutil import copyfile
from datetime import datetime
from models.resnet_model import get_resnet50_model
from utils.data_utils import get_data_loaders
from class_mapping import class_mapping  

init(autoreset=True)

# Add the root directory of the project to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir = os.path.join(project_root, 'config')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, latest_checkpoint_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, checkpoint_path)
    copyfile(checkpoint_path, latest_checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path} and {latest_checkpoint_path}")
    
    # Save model separately
    model_save_path = os.path.join(os.path.dirname(checkpoint_path), f'model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"{Fore.YELLOW}Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"{Fore.GREEN}Checkpoint loaded. Resuming from epoch {epoch + 1}")
        return epoch + 1, model, optimizer, loss
    else:
        print(f"{Fore.RED}No checkpoint found at '{checkpoint_path}'")
        return 0, model, optimizer, None

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, start_epoch=0):
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    for epoch in range(start_epoch, num_epochs):
        print(f"{Fore.CYAN}Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"{Fore.GREEN}Training", leave=False, ncols=100, ascii=True)
        
        for inputs, labels in train_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"{Fore.GREEN}Training Loss: {epoch_loss:.4f}")

        # Save checkpoint after training phase
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_train_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_path, latest_checkpoint_path)

        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        val_bar = tqdm(val_loader, desc=f"{Fore.BLUE}Validation", leave=False, ncols=100, ascii=True)

        with torch.no_grad():
            for inputs, labels in val_bar:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        print(f"{Fore.BLUE}Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n")

        # Save checkpoint after validation phase
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_val_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, latest_checkpoint_path)

    return model

def main():
    print(f"{Fore.YELLOW}Loading configuration...")
    # Configuration
    config_path = os.path.join(config_dir, 'config.yaml')
    if not os.path.isfile(config_path):
        print(f"{Fore.RED}Config file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"{Fore.GREEN}Configuration loaded successfully.")
    except Exception as e:
        print(f"{Fore.RED}Failed to load config file: {e}")
        sys.exit(1)

    print(f"{Fore.YELLOW}Getting data loaders...")
    try:
        train_loader, val_loader, num_classes = get_data_loaders(
            config['train_data_dir'], 
            config['val_data_dir'], 
            config['batch_size']
        )
        print(f"{Fore.GREEN}Data loaders ready.")
    except Exception as e:
        print(f"{Fore.RED}Error getting data loaders: {e}")
        sys.exit(1)

    print(f"{Fore.YELLOW}Initializing model...")
    try:
        model = get_resnet50_model(num_classes)
        print(f"{Fore.GREEN}Model initialized successfully.")
    except Exception as e:
        print(f"{Fore.RED}Error initializing model: {e}")
        sys.exit(1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

    # Load checkpoint if available
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    start_epoch, model, optimizer, _ = load_checkpoint(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'), model, optimizer)

    print(f"{Fore.YELLOW}Starting training...")
    # Start Training
    model = train_model(model, criterion, optimizer, train_loader, val_loader, config['num_epochs'], start_epoch)
    print(f"{Fore.GREEN}Training completed.")

if __name__ == "__main__":
    main()
