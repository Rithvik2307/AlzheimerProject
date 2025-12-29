import torch
import torch.nn as nn
import torch.nn.functional as F

class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        
        # --- Feature Extraction (The "Eye") ---
        # Layer 1: Input (1 channel) -> 32 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces dimensions by half

        # Layer 2: 32 -> 64 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Layer 3: 64 -> 128 filters
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # --- Classification (The "Brain") ---
        # Flattening calculation:
        # Input: 128x128 -> Pool1 -> 64x64 -> Pool2 -> 32x32 -> Pool3 -> 16x16
        # Final Tensor shape: (Batch, 128 channels, 16 height, 16 width)
        self.flatten_dim = 128 * 16 * 16
        
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 2) # Binary Output: [NonDemented, Demented]

    def forward(self, x):
        # Convolution -> ReLU Activation -> Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for the fully connected layers
        x = x.view(-1, self.flatten_dim)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x