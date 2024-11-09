import torch
from torch.utils.data import DataLoader
from .evaluation import evaluate_model, calculate_loss

def train_model(model, train_loader: DataLoader, val_loader: DataLoader, device: str, num_epochs: int) -> None:
    '''
    Train the model and evaluate after each epoch.

    Args:
        model: The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        device (str): Device to run the training on ('cpu' or 'cuda').
        num_epochs (int): Number of training epochs.
    '''
    optimizer = torch.optim.Adam(model.parameters())  # using Adam optimizer (usually used in LLMs)
    for epoch in range(num_epochs):
        model.train()  # setting the model to training mode
        total_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()  # forcing zero the gradients
            outputs = model(inputs)
            loss = calculate_loss(outputs, targets)
            loss.backward()  # setting backpropagation
            optimizer.step()  # updating weights
            total_loss += loss.item() * inputs.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss = evaluate_model(model, val_loader, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
