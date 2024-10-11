import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import seaborn as sns
import numpy as np
import os

import matplotlib.pyplot as plt

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
wgs84 = Proj(proj="latlong", datum="WGS84")
utm = Proj(proj="utm", zone=43, datum="WGS84")  # Adjust 'zone' based on your study area

coords_utm = []
for lat, lon in coords:
    x, y = transform(wgs84, utm, lon, lat)
    coords_utm.append((x, y))

coords_utm = np.array(coords_utm)


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


## Saving the model

save_path = '/content/drive/My Drive/MoNET_Thesis/gmm_model.pth'

# Save the model's state dictionary to the specified path
torch.save(model.state_dict(), save_path)

print(f"Model saved successfully at: {save_path}")
print("Model saved at:", os.path.abspath('gmm_model.pth'))



### From this part onwards, we will apply the model over the daily dataset to get rainfall surface for each days


# Separate the data by each unique date
unique_dates = dta['DATE.YYYY.MM.DD.'].unique()

# Store separated data in a dictionary
data_by_date = {}

for date in unique_dates:
    daily_data = dta[dta['DATE.YYYY.MM.DD.'] == date]
    coords = daily_data[['Latitude', 'Longitude']].values
    rainfall = daily_data['RAINFALL.DAILY.CUMULATIVE..0.5.mm.or.more.'].values

    coords_tensor = torch.tensor(coords, dtype=torch.float)
    rainfall_tensor = torch.tensor(rainfall, dtype=torch.float)

    data_by_date[date] = {
        'coords': coords_tensor,
        'rainfall': rainfall_tensor
    }

# Check the structure of data for a sample date
sample_date = unique_dates[21]
print(f"Data for {sample_date}:")
print(f"Coords: {data_by_date[sample_date]['coords'].shape}")
print(f"Rainfall: {data_by_date[sample_date]['rainfall'].shape}")

## Creates an edge index using k-nearest neighbors based on the coordinates.

def create_graph_from_coords(coords, k= 10):

    adj = kneighbors_graph(coords.numpy(), k, mode='distance', include_self=False)
    edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
    return edge_index

## The Prediction Grid

# Define bounds and grid resolution
lat_min, lat_max = 28.5, 31.5
lon_min, lon_max = 77.5, 81.0
resolution = 0.25  # Grid resolution(in degrees)

# Generate grid points
latitudes = np.arange(lat_min, lat_max + resolution, resolution)
longitudes = np.arange(lon_min, lon_max + resolution, resolution)

# Create a meshgrid for latitudes and longitudes
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Flatten the grid arrays
grid_coords = np.vstack([lat_grid.flatten(), lon_grid.flatten()]).T

# Convert to torch tensor
grid_coords_tensor = torch.tensor(grid_coords, dtype=torch.float)

##

# Function to create graph based on both grid and daily coordinates
def create_combined_graph(grid_coords, daily_coords, k=10):
    all_coords = torch.cat([grid_coords, daily_coords], dim=0)
    edge_index = create_graph_from_coords(all_coords, k=k)
    return edge_index, len(grid_coords)

# Dictionary to store grid predictions by date
grid_predictions_by_date = {}

for date in unique_dates:
    # Get the daily data
    daily_coords = data_by_date[date]['coords']
    daily_rainfall = data_by_date[date]['rainfall']

    # Combine grid and daily data into a single graph
    combined_edge_index, grid_size = create_combined_graph(grid_coords_tensor, daily_coords)

    # Create a data object with combined graph
    x_combined = torch.cat([grid_coords_tensor, daily_coords], dim=0)
    data_combined = Data(x=x_combined, edge_index=combined_edge_index)

    # Model inference
    with torch.no_grad():
        predictions_combined = model(data_combined)

    # Extract only the grid predictions
    grid_predictions = predictions_combined[:grid_size]

    # Convert predictions to numpy for plotting
    grid_predictions_numpy = grid_predictions.numpy()

    # Reshape predictions to match the grid shape
    predictions_grid = grid_predictions_numpy.reshape(len(latitudes), len(longitudes))

    # Store the predictions for later use
    grid_predictions_by_date[date] = predictions_grid

# Define the number of days to plot
num_days = len(grid_predictions_by_date)

# Define the number of rows and columns for the grid
num_rows = 4
num_cols = (num_days + num_rows - 1) // num_rows  # Compute columns needed to fit all days

# Create a figure and axes for the grid of plots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 4 * num_rows), constrained_layout=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Find the global minimum and maximum for the color scale
all_predictions = np.concatenate([grid_predictions.flatten() for grid_predictions in grid_predictions_by_date.values()])
vmin, vmax = np.min(all_predictions), np.max(all_predictions)

# Define the color palette
cmap = sns.color_palette("flare", as_cmap=True)

# Plot predictions for each day
for i, (date, predictions_grid) in enumerate(grid_predictions_by_date.items()):
    ax = axes[i]

    # Reshape predictions to match the grid shape
    predictions_grid = predictions_grid.reshape(len(latitudes), len(longitudes))

    # Plotting
    c = ax.pcolormesh(lon_grid, lat_grid, predictions_grid, cmap= cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Set the background to light grey
    plt.gca().set_facecolor('#e2e2e2')
    ax.set_title(f'{date}')

# Add a single colorbar for the entire figure
fig.colorbar(c, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1, label='Predicted Rainfall (mm)')

# Add overall title
fig.suptitle('Predicted Rainfall for Different Days', fontsize=16)

# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()