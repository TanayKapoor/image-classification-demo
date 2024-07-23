import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts.train import train_model, load_checkpoint
from models.resnet_model import get_resnet50_model
from utils.data_utils import get_data_loaders
from class_mapping import class_mapping  # Import the class mapping

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    
    # Load the configuration file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get data loaders for training and validation
    train_loader, val_loader, num_classes = get_data_loaders(
        config['train_data_dir'], 
        config['val_data_dir'], 
        config['batch_size']
    )

    # Initialize the model
    model = get_resnet50_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

    # Load checkpoint if available
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    start_epoch, model, optimizer, _ = load_checkpoint(latest_checkpoint, model, optimizer)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, config['num_epochs'], start_epoch)

if __name__ == '__main__':
    main()