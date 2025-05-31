import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from load_images import ImageLoad  # Assuming ImageLoad is saved in load_images.py
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class Convoultion_NN(ImageLoad):
    def __init__(self, dataset_path: str, learning_rate: float = 0.001, batch_size: int = 32, 
                 input_channels: int = 3, architecture: str = "deep-wide"):
        # Inherit from ImageLoad
        super().__init__(dataset_path)
        # Pre-Process Images
        self.image_paths = self._load_image_paths()
        self.main_loop()
        # Input shape defaults to (input_channels, height, width)
        self.input = (input_channels, self.size[0], self.size[1])
        self.number_of_labels = self.df['Label'].nunique()
        self.learning_rate = learning_rate
        self.input_channels = input_channels  
        # For speed/memory; changes epoch iterations as batch_size must complete dataset cycle
        self.batch_size = batch_size 
        self.architecture = architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Begin AI Pre-processing

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['EncodedLabel'] = self.label_encoder.fit_transform(self.df['Label'])
        self.number_of_labels = len(self.label_encoder.classes_)

        # Convert images and labels to tensors
        self.image_tensors, self.label_tensors = self.create_image_tensors()
        self.save_tensors()
        # Define model and parameters
        self.model = self.build_model(architecture=self.architecture).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def create_image_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts image data from the DataFrame into tensors and normalizes them.

        Returns:
            Tuple of image and label tensors.
        """
        image_tensors = []
        label_tensors = torch.tensor(self.df['EncodedLabel'].values, dtype=torch.long).to(self.device)
        print('🦊 creating image and label tensors 🦊')
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            # Fixes copying error on local machine
            # TODO: Change for deployment
            img_array = row['Image'].copy()  
            img_tensor = torch.tensor(img_array, dtype=torch.float32) #/ 255.0  # Normalize to [0, 1]
            #img_tensor = F.normalize(img_tensor) 
            img_tensor = img_tensor.permute(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
            image_tensors.append(img_tensor)

        image_tensors = torch.stack(image_tensors).to(self.device)
        print('Image and label tensors created')
        return image_tensors, label_tensors
    
    def build_model(self, architecture: str) -> nn.Sequential:
        """
        Builds the CNN model based on the specified architecture.

        Args:
            architecture (str): Architecture type ("deep-wide").

        Returns:
            nn.Sequential: A sequential model based on the desired architecture.
        """
        layers = []

        if architecture == "deep-wide":
            # Wide and deep architecture

            layers = [
                # Convolution Block 1
                nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.2),

                # Convolution Block 2
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),  # Corrected to 128 to match Conv2d output channels
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.3),

                # Convolution Block 3
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.5),

                # Global Average Pooling
                nn.AdaptiveAvgPool2d(1),  # Output will be 256 x 1 x 1

                # Fully Connected Layers
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(0.4),

                # Output Layer
                nn.Linear(128, self.number_of_labels)
            ]
        else:
            raise ValueError(f"Unsupported architecture type '{architecture}'")

        # Wrap layers in Sequential
        self.model = nn.Sequential(*layers)

        return self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.model(x)
    
    def train_model(self, epochs: int):
        """
        Trains the CNN model for a specified number of epochs.

        Args:
            epochs (int): Number of training epochs.
        """
        # Set the model to training mode
        self.model.train()

        # Convert the image_tensors and label_tensors to numpy arrays for splitting
        images = self.image_tensors.numpy()  
        labels = self.label_tensors.numpy()

        # Perform train-test split (80% training, 20% testing)
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Convert back to torch tensors
        train_images = torch.tensor(train_images)
        test_images = torch.tensor(test_images)
        train_labels = torch.tensor(train_labels)
        test_labels = torch.tensor(test_labels)

        # Create TensorDataset for train and test data
        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)

        # Create DataLoader for train and test datasets
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Iterate over epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()  # Ensure the model is in training mode

            # Training loop
            for img_tensors, label_tensors in tqdm(train_dataloader, total=len(train_dataloader)):
                img_tensors, label_tensors = img_tensors.to(self.device), label_tensors.to(self.device)
                
                # Forward pass
                output = self.model(img_tensors)
                loss = self.loss_function(output, label_tensors)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Backpropagation
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
            torch.save(self.model.state_dict(), f"model_epoch_{epoch + 1}.pth")
            
        self.model.eval()  # Set the model to evaluation mode
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():  # No gradients needed for evaluation
            for img_tensors, label_tensors in test_dataloader:
                img_tensors, label_tensors = img_tensors.to(self.device), label_tensors.to(self.device)
                
                # Forward pass
                output = self.model(img_tensors)
                
                # Get predictions (e.g., using argmax for classification)
                _, predicted = torch.max(output, 1)
                
                # Track the number of correct predictions
                total_samples += label_tensors.size(0)
                total_correct += (predicted == label_tensors).sum().item()

        # Calculate test accuracy after each epoch
        accuracy = 100 * total_correct / total_samples
        print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%")
        
        # Optionally save model checkpoints after each epoch (if needed)
        # torch.save(self.model.state_dict(), f"model_epoch_{epoch + 1}.pth")

    def process_image(self, file_path: str) -> str:
        """
        Processes and predicts the label for a single image using the trained model.

        Args:
            file_path (str): Path to the image file.

        Returns:
            str: Predicted label.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # Disable gradient calculation for inference
            img_tensor = self.image_to_tensor(file_path)  # Convert image to tensor
            output = self.model(img_tensor.unsqueeze(0))  # Forward pass (add batch dimension)
            
            # Get the predicted label by finding the index of the max log-probability
            predicted_label_index = torch.argmax(output.data, dim=1).item()
            predicted_label = self.label_encoder.inverse_transform([predicted_label_index])[0]  # Convert back to label
            
            print("Model output:", output)  # Print the raw output for debugging
            return predicted_label
        
    def image_to_tensor(self, file_path: str = None, numpy_array: np.ndarray = None) -> torch.Tensor:
        """
        Turn numpy array or file_path into Tensor with normalized pixel values.

        Args:
            file_path (str, optional): Path to the image file.
            numpy_array (np.ndarray, optional): Numpy array representing the image.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        if file_path:
            image = self._open_img(file_path, add_noise=False)
        else:
            image = numpy_array

        #image = image.copy()
        img_tensor = torch.tensor(image, dtype=torch.float32) #/ 255.0  # Normalize to [0, 1]
        #img_tensor = F.normalize(img_tensor)  # Further normalization (optional)

        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img_tensor.to(self.device)  # Return tensor on correct device

    def save_tensors(self):
        # Save tensors to file
        torch.save(self.image_tensors, "image_tensors.pt")
        torch.save(self.label_tensors, "label_tensors.pt")
        print("Tensors saved to disk.")

    def test_model(self, test_image_tensors: torch.Tensor=None, test_label_tensors: torch.Tensor = None) -> float:
        """
        Tests the CNN model on a set of test images.

        Args:
            test_image_tensors (torch.Tensor): Tensors of test images.
            test_label_tensors (torch.Tensor): Tensors of true labels for the test images.

        Returns:
            float: Test accuracy as a percentage.
        """
        self.model.eval()  # Set the model to evaluation mode
        correct = 0  # Counter for correct predictions
        total = len(self.label_tensors) # Total number of test samples
        print('Total:',total)
        with torch.no_grad():  # Disable gradient computation for testing
            for i in tqdm(range(total)):
                img_tensor = self.image_tensors[i].unsqueeze(0).to(self.device)  # Add batch dimension
                label = self.label_tensors[i].to(self.device)
                output = self.model(img_tensor)  # Forward pass

                # Get the predicted label (index with max probability)
                predicted_label_index = torch.argmax(output, dim=1).item()
                
                # Check if the prediction matches the true label
                if predicted_label_index == label.item():
                    correct += 1
        
            accuracy = (correct / total) * 100
            print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy


if __name__ == "__main__":
    folder_path = "~/tricorder/data/"
    print('Initializing image loader...')
    images = ImageLoad(folder_path)
    # Prepare images for neural network
    images.main_loop()
    df = images.df  # Assuming df is now populated with image data
