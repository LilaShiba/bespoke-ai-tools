import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Tuple

class NN(nn.Module):
    """
    Customizable time series neural network with random activation layers.
    
    Args:
        input_size (int): Number of input features.
        hidden_sizes (List[int]): List of sizes for each hidden layer.
        output_size (int): Number of output features.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__() # torch.model 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Define layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(self._get_random_activation())
            prev_size = size
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def _get_random_activation(self) -> nn.Module:
        """
        Returns a random activation function from a predefined set.
        
        Returns:
            nn.Module: Activation function instance.
        """
        activations = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.ELU()]
        return activations[torch.randint(len(activations), (1,)).item()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def train_model(self,
                    train_loader: torch.utils.data.DataLoader,
                    epochs: int = 10,
                    lr: float = 0.001,
                    device: Optional[torch.device] = None) -> None:
        """
        Train the model on provided data.
        
        Args:
            train_loader (DataLoader): Training data loader.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            device (torch.device, optional): Device to run training on.
        """
        device = device or torch.device('cpu')
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def test_model(self,
                   test_loader: torch.utils.data.DataLoader,
                   device: Optional[torch.device] = None) -> float:
        """
        Test the model and compute average loss.
        
        Args:
            test_loader (DataLoader): Test data loader.
            device (torch.device, optional): Device to run testing on.
        
        Returns:
            float: Average test loss.
        """
        device = device or torch.device('cpu')
        self.to(device)
        criterion = nn.MSELoss()
        
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss
