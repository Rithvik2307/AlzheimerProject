import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_data_loaders(data_dir='./Dataset', batch_size=32):
    # Transformation Pipeline
    # 1. Grayscale: MRI scans are single channel.
    # 2. Resize: Standardize to 128x128 for the CNN.
    # 3. ToTensor: Convert pixels (0-255) to FloatTensor (0.0-1.0).
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load Data
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' not found. Please check your dataset path.")

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split Data (80% Train, 20% Validation)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size])

    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data Loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images.")
    print(f"Classes: {full_dataset.classes}")
    
    return train_loader, val_loader