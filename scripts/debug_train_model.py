import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet_model import get_resnet50_model
from utils.data_utils import get_data_loaders

# Load configuration
import yaml
with open('./config/config.yaml') as f:
    config = yaml.safe_load(f)

# Get data loaders
train_loader, val_loader, num_classes = get_data_loaders(
    config['train_data_dir'], 
    config['val_data_dir'], 
    config['batch_size']
)

# Initialize the model, criterion, optimizer
model = get_resnet50_model(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

# Debugging function
def train_model_debug(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    print(f"Number of classes: {num_classes}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Debugging: Print the size of inputs
            print(f"Batch {i}, Input size: {inputs.size()}")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f}")

    return model

# Start training
train_model_debug(model, criterion, optimizer, train_loader, val_loader, num_epochs=5)
