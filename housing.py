import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

def load_data(file_name):
    file_path = os.path.abspath(file_name)
    data = pd.read_csv(file_path)
    return data

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    # print(data.head())
    # print(data.info())
    # print(data.describe())
    # print(data['ocean_proximity'].value_counts())
    longitude, latitude, median_income = df['longitude'].values, df['latitude'].values, df['median_income'].values
    std_longitude, std_latitude = np.std(longitude), np.std(latitude)
    mean_longitude, mean_latitude = np.mean(longitude), np.mean(latitude)
    income_mean = median_income.mean()
    income_std  = median_income.std()
    ocean_proximity = df['ocean_proximity'].values
    
    Y = df['median_house_value'].values.reshape(-1, 1)
    Y = Y * 1.56 # Scaling by average house price increase over past 8 years
    Y_mean = Y.mean()
    Y_std = Y.std()
    norm_stats = {
        'Y_mean': Y_mean,
        'Y_std': Y_std,
        'lon_mean': mean_longitude,
        'lon_std': std_longitude,
        'lat_mean': mean_latitude,
        'lat_std': std_latitude,
        'income_mean': income_mean,
        'income_std': income_std
    }
    
    # Normalizing
    longitude = (longitude - mean_longitude) / std_longitude
    latitude = (latitude - mean_latitude) / std_latitude
    median_income = (median_income - income_mean) / income_std
    X = np.column_stack((longitude, latitude, median_income))
    Y = (Y - Y_mean) / Y_std
    
    return df, X, Y, norm_stats

class HousingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)
        

def predict_price(model, lat, lon, income, stats):
    # normalize inputs
    lon_norm = (lon - stats['lon_mean']) / stats['lon_std']
    lat_norm = (lat - stats['lat_mean']) / stats['lat_std']
    income_norm = (income - stats['income_mean']) / stats['income_std']
    inp = torch.tensor([[lon_norm, lat_norm, income_norm]], dtype=torch.float32)

    with torch.no_grad():  # no gradients during inference
        pred_norm = model(inp)

    # un-normalize output
    pred_price = pred_norm.item() * stats['Y_std'] + stats['Y_mean']
    return pred_price


def main():
    df = load_data('housing.csv')
    df, X, Y, norm_stats = preprocess_data(df)
    df.shape

    # Before starting on regression, I want to see if I can preemptively identify correlation with 
    # a heatmap, excluding categorical features
    df_numeric = df.select_dtypes(include=[np.number, np.float64, np.int64])
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f")
    plt.show()
    
    # Previewing correlation shows little relevance between long / lat and median house value,
    # Though we might see better results with nonlinear regression.
    # However, we do see strong correlation with number of rooms and bedrooms. Insightful.
    
    # Starting regression model
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    # Scaling down longitude and latitude
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True) # batching
    
    model = HousingModel()
    
    criterion = nn.MSELoss() # MSE Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(500):
        epochLoss = 0.0
        for batch_X, batch_Y in dataLoader:
            # Forward Pass
            predictions = model(batch_X)
            # Loss
            loss = criterion(predictions, batch_Y)
            # zero gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            
            epochLoss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epochLoss/len(dataLoader)}")
    
    # Evaluation
    model.eval()
    test_lon = -122.42183900
    test_lat = 37.77378200
    avg_income = 96.421
    price = predict_price(model, test_lat, test_lon, avg_income, stats=norm_stats)
    print("Estimated price:", price.floor())


if __name__ == "__main__":
    main()