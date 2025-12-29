import torch
from dataset import get_data_loaders
from model import AlzheimerCNN
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = get_data_loaders()  
model = AlzheimerCNN().to(device)

# 2. Load the best brain
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
except FileNotFoundError:
    print("Error: 'best_model.pth' not found.")
    exit()

model.eval()

# 3. Gather all predictions
all_preds = []
all_labels = []

print("Running detailed evaluation...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. The Moment of Truth
# Print the Classification Report (Precision, Recall, F1-Score)
print("\n--- Detailed Report ---")
print(classification_report(all_labels, all_preds, target_names=['Demented', 'NonDemented']))

# Print Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n--- Confusion Matrix ---")
print(f"True Negatives (Correctly Healthy): {cm[1][1]}")
print(f"False Positives (Healthy but flagged Demented): {cm[1][0]}")
print(f"False Negatives (Demented but missed): {cm[0][1]}  <-- MOST DANGEROUS")
print(f"True Positives (Correctly Demented): {cm[0][0]}")