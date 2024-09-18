import os
import pandas as pd
import numpy as np
import re
import json
from torch.utils.data import DataLoader, Dataset
import torch

from dataloader import loader, dataset

import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
        return x

def train_model():
    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Early stopping parameters based on median q-error
    patience = 30  # Number of epochs with no improvement after which training will be stopped
    best_qerror = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Training loop with early stopping based on median q-error
    num_epochs = 10000
    model.train()
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_losses = []
            median_qerrors = []
            for batch in val_loader:
                out = model(batch)
                val_loss = criterion(out.view(-1), batch.y)
                val_losses.append(val_loss.item())
                median_qerror = qerror(batch.y.numpy(), out.view(-1).numpy())
                median_qerrors.append(median_qerror)
            avg_val_loss = np.mean(val_losses)
            avg_median_qerror = np.mean(median_qerrors)
            print(f"Median Qerror on Validation Set: {avg_median_qerror:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Early stopping check based on median q-error
        if avg_median_qerror < best_qerror:
            best_qerror = avg_median_qerror
            epochs_no_improve = 0
            print(f"epochs_no_improve: {epochs_no_improve}")
            # Save the best model
            torch.save(model.state_dict(), 'best_peak_memory_gnn_model.pth')
        else:
            epochs_no_improve += 1
            print(f"epochs_no_improve: {epochs_no_improve}")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop = True

        # Save the last model regardless
        torch.save(model.state_dict(), 'peak_memory_gnn_model.pth')


# Initialize the model
num_node_features = dataset[0].x.shape[1]  # Ensure all x have the same feature dimension
model = PeakMemoryGNN(num_node_features=num_node_features, hidden_dim=128, heads=4)


from sklearn.model_selection import train_test_split
train_loader, val_loader = train_test_split(dataset, test_size=0.2, random_state=seed)



def qerror(y_true, y_pred):
    epsilon = 1e-10
    y_pred_safe = np.where(y_pred <= 0, epsilon, y_pred)
    y_true_safe = np.where(y_true <= 0, epsilon, y_true)
    qerror = np.maximum(y_pred_safe / y_true_safe, y_true_safe / y_pred_safe)
    return np.percentile(qerror, 50)

skip_train = True
if not skip_train:
    train_model()

# Test the model on the test set
# reload the best model and test
model.load_state_dict(torch.load('best_peak_memory_gnn_model_graphtransformer.pth'))
model.eval()
with torch.no_grad():
    test_losses = []
    median_qerrors = []
    for batch in val_loader:
        out = model(batch)
        # test_loss = criterion(out.view(-1), batch.y)
        # test_losses.append(test_loss.item())
        median_qerror = qerror(batch.y.numpy(), out.view(-1).numpy())
        median_qerrors.append(median_qerror)
    avg_test_loss = np.mean(test_losses)
    avg_median_qerror = np.mean(median_qerrors)
    print(f"Median Qerror on Test Set: {avg_median_qerror:.4f}")
