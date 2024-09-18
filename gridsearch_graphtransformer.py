import os
import pandas as pd
import numpy as np
import re
import json
from torch.utils.data import random_split
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from itertools import product
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric's DataLoader
from dataloader import dataset
# Set random seeds for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Assume 'dataset' is already loaded and preprocessed
# Replace this with your actual dataset loading code
# For example:
# from dataloader import loader, dataset
# dataset = loader.load_dataset()

# Placeholder for dataset
# Replace this with your actual dataset
# Example:
# dataset = YourPreprocessedDataset()

# For demonstration, let's assume 'dataset' is already defined
# Ensure that 'dataset' is a list of torch_geometric.data.Data objects
# If using a custom dataset, ensure it inherits from torch_geometric.data.Dataset

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

# Create DataLoaders using PyTorch Geometric's DataLoader
batch_size = 32  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the GNN model with TransformerConv
class PeakMemoryGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, heads=4):
        super(PeakMemoryGNN, self).__init__()
        self.conv1 = TransformerConv(num_node_features, hidden_dim, heads=heads)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.conv3 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.fc1 = torch.nn.Linear(hidden_dim * heads, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # Output is a single value

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global Pooling (e.g., mean pooling)
        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(-1)  # Ensure output shape matches target

def qerror(y_true, y_pred):
    epsilon = 1e-10
    y_pred_safe = np.where(y_pred <= 0, epsilon, y_pred)
    y_true_safe = np.where(y_true <= 0, epsilon, y_true)
    qerror_values = np.maximum(y_pred_safe / y_true_safe, y_true_safe / y_pred_safe)
    return np.median(qerror_values)

def train_model(model, train_loader, val_loader, device, learning_rate=0.001, weight_decay=0, num_epochs=1000, patience=30):
    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_qerror_val = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to device
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())  # Ensure targets are float
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_losses = []
            qerrors = []
            for batch in val_loader:
                batch = batch.to(device)  # Move batch to device
                out = model(batch)
                val_loss = criterion(out, batch.y.float())
                val_losses.append(val_loss.item())
                median_qerror = qerror(batch.y.cpu().numpy(), out.cpu().numpy())
                qerrors.append(median_qerror)
            avg_val_loss = np.mean(val_losses)
            avg_median_qerror = np.mean(qerrors)
            print(f"Validation Loss: {avg_val_loss:.4f}, Median Qerror on Validation Set: {avg_median_qerror:.4f}")

        # Early stopping check based on median q-error
        if avg_median_qerror < best_qerror_val:
            best_qerror_val = avg_median_qerror
            epochs_no_improve = 0
            print(f"Improved validation q-error to {best_qerror_val:.4f}")
            # Save the best model
            torch.save(model.state_dict(), 'best_peak_memory_gnn_model.pth')
        else:
            epochs_no_improve += 1
            print(f"No improvement in q-error for {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                early_stop = True

        # Optionally, save the last model
        torch.save(model.state_dict(), 'peak_memory_gnn_model.pth')

    return best_qerror_val

# Hyperparameter grid
param_grid = {
    'learning_rate': [0.01],
    'weight_decay': [1e-4],
    'hidden_dim': [64, 128, 256],
    'heads': [2, 4, 8],
}

# Generate all combinations of hyperparameters
param_combinations = list(product(
    param_grid['learning_rate'],
    param_grid['weight_decay'],
    param_grid['hidden_dim'],
    param_grid['heads']
))

best_params = None
best_qerror = float('inf')

# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Starting grid search over hyperparameters...")
for idx, (lr, wd, hd, h) in enumerate(param_combinations, 1):
    print(f"\nTesting combination {idx}/{len(param_combinations)}:")
    print(f"Learning Rate: {lr}, Weight Decay: {wd}, Hidden Dim: {hd}, Heads: {h}")

    # Initialize the model with current hyperparameters
    num_node_features = dataset[0].x.shape[1]  # Ensure all x have the same feature dimension
    model = PeakMemoryGNN(num_node_features=num_node_features, hidden_dim=hd, heads=h)
    model = model.to(device)

    # Train the model
    median_qerror_val = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        learning_rate=lr,
        weight_decay=wd,
        num_epochs=1000,
        patience=30
    )

    print(f"Median Qerror on Validation Set: {median_qerror_val:.4f}")

    # Update best parameters if current model is better
    if median_qerror_val < best_qerror:
        best_qerror = median_qerror_val
        best_params = {
            'learning_rate': lr,
            'weight_decay': wd,
            'hidden_dim': hd,
            'heads': h
        }
        # Save the best model
        torch.save(model.state_dict(), 'best_peak_memory_gnn_model_grid_search.pth')
        print(f"New best q-error: {best_qerror:.4f} with parameters {best_params}")

print("\nGrid search complete.")
print(f"Best Median Qerror: {best_qerror:.4f}")
print("Best Hyperparameters:")
print(best_params)

# Load the best model and evaluate on the validation set
best_model = PeakMemoryGNN(num_node_features=num_node_features, hidden_dim=best_params['hidden_dim'], heads=best_params['heads'])
best_model.load_state_dict(torch.load('best_peak_memory_gnn_model_grid_search.pth'))
best_model = best_model.to(device)
best_model.eval()

with torch.no_grad():
    qerrors = []
    for batch in val_loader:
        batch = batch.to(device)
        out = best_model(batch)
        median_qerror = qerror(batch.y.cpu().numpy(), out.cpu().numpy())
        qerrors.append(median_qerror)
    final_median_qerror = np.mean(qerrors)
    print(f"\nMedian Qerror on Validation Set with Best Model: {final_median_qerror:.4f}")
