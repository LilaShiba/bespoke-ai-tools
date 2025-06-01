import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from sp00kyVectors.neural_net import NN  # Replace with your actual filename

def create_synthetic_data(num_samples=100, input_size=5, output_size=1):
    X = torch.randn(num_samples, input_size)
    weights = torch.randn(input_size, output_size)
    y = X @ weights + 0.1 * torch.randn(num_samples, output_size)
    return X, y

@pytest.fixture
def model():
    return NN(input_size=5, hidden_sizes=[10, 5], output_size=1)

@pytest.fixture
def dataloaders():
    X, y = create_synthetic_data(100, 5, 1)
    split = 80
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)
    return train_loader, test_loader

def test_training_and_testing(model, dataloaders):
    train_loader, test_loader = dataloaders
    
    # Capture initial loss
    model.train()
    initial_loss = None
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        initial_loss = loss.item()
        break
    
    # Train for 1 epoch
    model.train_model(train_loader, epochs=1, lr=0.01)
    
    # Check model outputs shape
    for inputs, targets in test_loader:
        outputs = model(inputs)
        assert outputs.shape == targets.shape
        break
    
    # Run test method to check no errors
    model.test_model(test_loader)
    
    # Check loss decreased after training
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            total_loss += loss.item()
            count += 1
    avg_loss = total_loss / count
    
    assert avg_loss <= initial_loss, "Loss did not decrease after training"
