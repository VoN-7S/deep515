import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms

from models.MNISTModels import MLPBase
from parameters import TrainingConfig, SystemConfig


def get_loaders(training_config: TrainingConfig, system_config: SystemConfig) -> tuple[DataLoader, DataLoader]:

    """
    Creates and returns training and validation DataLoaders for the MNIST dataset.

    Args:
        training_config (TrainingConfig): Contains batch size and number of workers.
        system_config (SystemConfig): Contains dataset download path.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders respectively.
    """


    # Normalize the datasets.
    ds_transform = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)
    ))
    # Load the datasets.
    training_ds = datasets.MNIST(
        root=system_config.dataset_path,
        train=True,
        transform=ds_transform,
        download=True
    )

    val_ds = datasets.MNIST(
        root=system_config.dataset_path,
        train=False,
        transform=ds_transform,
        download=True
    )
    # Sample all 60k training indices — modify range to use a subset.
    indices = list(range(60000))
    sampler = SubsetRandomSampler(indices)
    # Prepare the dataloaders.
    training_dl = DataLoader(training_ds, training_config.batch_size, num_workers=training_config.num_workers, sampler=sampler)
    val_dl = DataLoader(val_ds, training_config.batch_size, shuffle=False, num_workers=training_config.num_workers)

    return training_dl, val_dl

def train_one_epoch(training_config: TrainingConfig, model: MLPBase, lossf: nn.modules.loss, optimizer: torch.optim, device: str, train_loader: DataLoader) -> tuple[float, float]:

    """
    Trains the model for one epoch over the training set.

    Args:
        training_config (TrainingConfig): Training configurations.Contains batch size, log interval and regularization settings.
        model (MLPBase): The MLP model to train.
        lossf (nn.modules.loss): Loss function.
        optimizer (torch.optim): Optimizer for updating model parameters.
        device (str): Device to run the model on (e.g. 'cpu', 'cuda').
        train_loader (DataLoader): DataLoader for the training set.

    Returns:
        tuple[float, float]: Average loss and accuracy over the epoch.
    """

    model.train() # Convert the model to the training mode (e.g. dropout).
    total_loss = 0
    acc = 0

    for batch_idx, (batch, label_idx) in enumerate(train_loader, 0):
        
        
        optimizer.zero_grad() # Reset the gradients to avoid accumulation.
        pred = model(batch.to(device))
        loss = lossf(pred, label_idx)
        l_loss = loss + get_regularization_term(training_config, model) # Add regularization term on top of base loss.

        l_loss.backward() # Activate backward automatic differentiation and track the gradients.
        optimizer.step() # Update the weights.


        total_loss += loss.detach().item()
        # Accumulate correctly classified samples.
        acc += torch.sum(torch.eq(torch.argmax(pred.detach(), 1),label_idx.to(device))).item()

        if (batch_idx + 1) % training_config.log_interval == 0:
            print(f"\nBatch #{batch_idx + 1}")
            print(f"Average Loss: {total_loss / ((batch_idx + 1) * training_config.batch_size): .4f} | Average Accuracy: {acc / ((batch_idx + 1) * training_config.batch_size): .4f}")
    

    return total_loss / len(train_loader.dataset), acc / len(train_loader.dataset)

def validate(model: MLPBase, lossf: nn.modules.loss, device: str, val_loader: DataLoader) -> tuple[float, float]:

    """
    Evaluates the model on the validation set.

    Args:
        model (MLPBase): The MLP model to evaluate.
        lossf (nn.modules.loss): Loss function.
        device (str): Device to run the model on (e.g. 'cpu', 'cuda').
        val_loader (DataLoader): DataLoader for the validation set.

    Returns:
        tuple[float, float]: Average loss and accuracy over the validation set.
    """

    model.eval() # Convert the model to the evaluation mode (e.g. dropout layer).
    total_loss = 0 # Total loss.
    acc = 0 # Accurate predictions.

    # Disable gradient computation for efficiency during validation
    with torch.no_grad():

        for batch_idx, (batch, label_idx) in enumerate(val_loader, 0): 
            pred = model(batch.to(device))
            loss = lossf(pred, label_idx)
            # Accumulate loss and accuracy.
            total_loss += loss.detach().item()
            acc += torch.sum(torch.eq(torch.argmax(pred.detach(), 1),label_idx)).item()
            


    return total_loss / len(val_loader.dataset), acc / len(val_loader.dataset)

def get_regularization_term(training_config: TrainingConfig, model) -> torch.Tensor:

    """
    Computes the regularization term for the model parameters.

    Supports both L1 (regularizer=1) and L2 (regularizer=2) regularization
    controlled by training_config.regularizer.

    Args:
        training_config (TrainingConfig): Contains weight decay and regularizer type.
        model: The model whose parameters are regularized.

    Returns:
        torch.Tensor: Regularization term to be added to the loss.
    """

    return training_config.weight_decay * sum([torch.sum(torch.abs(p) ** training_config.regularizer) for p in model.parameters()]) # Loop through all parameters, take the absolute value and norm, then summ all.

def set_seed(seed: int):

    """
    Sets the random seed across all libraries to ensure reproducibility.

    Args:
        seed (int): Seed value to use.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False







