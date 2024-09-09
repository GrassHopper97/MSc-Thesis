import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import seaborn as sns
import numpy as np

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv

## The Dataset

drive.mount('/content/drive')
dataset = "/content/drive/My Drive/MoNET_Thesis/July_2023_with_elev"
one_day = "/content/drive/My Drive/MoNET_Thesis/July_2023_with_elev/processed_4_July_23.csv"
all_day =  "/content/drive/My Drive/MoNET_Thesis/Modified_again_july.csv"

dta = pd.read_csv(all_day)

## Data with the unique locations
# Select unique observations by dropping duplicates
unique_data = dta.drop_duplicates(subset=['Latitude', 'Longitude'])

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
from pyproj import Proj, transform
from sklearn.neighbors import kneighbors_graph


## Data Preparation
# We will use the columns `Latitude`, `Longitude`, and `RAINFALL.DAILY.CUMULATIVE..0.5.mm.or.more.`

# Extract the features (latitude, longitude) and target (rainfall)
coords = unique_data[['Latitude', 'Longitude']].values
rainfall = unique_data['RAINFALL.DAILY.CUMULATIVE..0.5.mm.or.more.'].values

# # Define the projection systems: WGS84 (lat/lon) and UTM
# wgs84 = Proj(proj="latlong", datum="WGS84")
# utm = Proj(proj="utm", zone=43, datum="WGS84")  # Adjust 'zone' based on your study area

# coords_utm = []
# for lat, lon in coords:
#     x, y = transform(wgs84, utm, lon, lat)
#     coords_utm.append((x, y))

# coords_utm = np.array(coords_utm)


# Convert to PyTorch tensors
coords_tensor = torch.tensor(coords, dtype=torch.float)
rainfall_tensor = torch.tensor(rainfall, dtype=torch.float)

## Graph Construction and Generating Pytorch Geometric data

# Create edge indices using a simple distance threshold 

k = 7  # Number of neighbors
adj_matrix = kneighbors_graph(coords_tensor, n_neighbors=k, mode='distance', include_self=False)

# Create edge index from adjacency matrix
edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj_matrix)[0]

# Create a Data object for PyTorch Geometric
data = Data(x=coords_tensor, edge_index=edge_index, y=rainfall_tensor)


#### Train and Testing Sample Generation with Masks

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split the data
train_idx, val_idx = train_test_split(range(coords_tensor.shape[0]), test_size=0.2, random_state=42)

# Create masks for training and validation
train_mask = torch.zeros(coords_tensor.shape[0], dtype=torch.bool)
val_mask = torch.zeros(coords_tensor.shape[0], dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True

# Add masks to data object
data.train_mask = train_mask
data.val_mask = val_mask

# Print the masks
print(f'Training Mask Shape: {train_mask.shape}')
print(f'Validation Mask Shape: {val_mask.shape}')


#### Model Set Up

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv

class GMMNet(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size):
        super(GMMNet, self).__init__()
        self.conv1 = GMMConv(in_channels, 16, dim=dim, kernel_size=kernel_size)
        self.conv2 = GMMConv(16, out_channels, dim=dim, kernel_size=kernel_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Calculate pseudo-coordinates (difference in lat/long for edges)
        row, col = edge_index
        pseudo = x[row] - x[col]  # Relative positions

        print(f'Pseudo-coordinates Shape: {pseudo.shape}')

        x = F.elu(self.conv1(x, edge_index, pseudo))
        print(f'Output after conv1 Shape: {x.shape}')

        x = self.conv2(x, edge_index, pseudo)
        print(f'Output after conv2 Shape: {x.shape}')

        return x

# Initialize the model
model = GMMNet(in_channels=2, out_channels=1, dim=2, kernel_size=3)

# Print the model
print(model)

# Assume `data` has been set up with x, edge_index, and y as in previous steps
# Forward pass through the model to see the outputs
output = model(data)
print(f'Final Output Shape: {output.shape}')


#### Training Data

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.MSELoss()


# Lists to store loss values for plotting
train_losses = []
val_losses = []
train_r2_scores = []
val_r2_scores = []

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].unsqueeze(1))
    loss.backward()
    optimizer.step()

    # Calculate R² score for training data
    train_r2 = r2_score(data.y[data.train_mask].cpu().numpy(), out[data.train_mask].cpu().detach().numpy())

    return loss.item(), train_r2

#### Validation Data

def validate(data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask].unsqueeze(1))

        # Calculate R² score for validation data
        val_r2 = r2_score(data.y[data.val_mask].cpu().numpy(), out[data.val_mask].cpu().numpy())

    return val_loss.item(), val_r2

# Training loop
for epoch in range(140):  # Adjust the number of epochs as needed
    loss, train_r2 = train(data)
    train_losses.append(loss)
    train_r2_scores.append(train_r2)

    val_loss, val_r2 = validate(data)
    val_losses.append(val_loss)
    val_r2_scores.append(val_r2)

    if epoch % 2 == 0:
        print(f'Epoch {epoch}, Loss: {loss}, Train R²: {train_r2}, Val Loss: {val_loss}, Val R²: {val_r2}')

