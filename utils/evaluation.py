import torch
from torch.utils.data import DataLoader

def evaluate_model(model, dataloader: DataLoader, device: str) -> float:
    '''
    Evaluate the model on the given DataLoader.

    Args:
        model: The model to evaluate.
        dataloader (DataLoader): DataLoader containing the validation data.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        float: Average loss on the validation dataset.
    '''
    model.eval()  # setting the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # disabling gradient calculation for evaluation
        for batch in dataloader:
            inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            loss = calculate_loss(outputs, targets)  # Replace with your loss calculation method
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else float('inf')

def calculate_loss(outputs, targets) -> float:
    '''
    Calculate the loss between model outputs and targets.

    Args:
        outputs: The outputs from the model.
        targets: The target labels.

    Returns:
        float: Calculated loss.
    '''
    # implements loss function calculation
    loss_function = torch.nn.CrossEntropyLoss()
    return loss_function(outputs.view(-1, outputs.size(-1)), targets.view(-1))
