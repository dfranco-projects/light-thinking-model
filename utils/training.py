import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

def load_text_data(dataset_name: str):
    '''
    Load text dataset from Hugging Face.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        dataset: Loaded dataset.
    '''
    return load_dataset(dataset_name)

def train_model(model: nn.Module, dataloader, criterion: nn.Module, optimizer, num_epochs: int) -> None:
    '''
    Train the model.

    Args:
        model (nn.Module): The neural network model.
        dataloader: DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer: Optimizer for updating model weights.
        num_epochs (int): Number of epochs to train.
    '''
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
