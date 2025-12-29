import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import AlzheimerCNN

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train():
    # 1. Setup
    # Use MPS (Metal Performance Shaders) for Mac if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Training on: {device}")
    
    train_loader, val_loader = get_data_loaders(batch_size=BATCH_SIZE)
    model = AlzheimerCNN().to(device)
    
    # 2. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # 4. Save the model
    torch.save(model.state_dict(), "best_model.pth")
    print("Training Complete. Model saved as 'best_model.pth'")

if __name__ == "__main__":
    train()