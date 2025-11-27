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
    median_income = median_income * 1.233 # Scaling for modern income levels
    income_mean = median_income.mean()
    income_std  = median_income.std()
    ocean_proximity = df['ocean_proximity'].values
    # Modifying ocean_proximity to numerical values
    proximity_map = {
        '<1H OCEAN': 0,
        'INLAND': 1,
        'NEAR OCEAN': 2,
        'NEAR BAY': 3,
        'ISLAND': 4
    }
    ocean_proximity_num = np.array([proximity_map[val] for val in ocean_proximity])
    
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
    X = np.column_stack((longitude, latitude, median_income, ocean_proximity_num))
    Y = (Y - Y_mean) / Y_std
    
    return df, X, Y, norm_stats

class HousingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
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
        

def predict_price(model, lat, lon, income, prox, stats):
    # normalize inputs
    lon_norm = (lon - stats['lon_mean']) / stats['lon_std']
    lat_norm = (lat - stats['lat_mean']) / stats['lat_std']
    income_norm = (income - stats['income_mean']) / stats['income_std']
    inp = torch.tensor([[lon_norm, lat_norm, income_norm, prox]], dtype=torch.float32)
    inp = inp.to(next(model.parameters()).device)  # move to same device as model

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
    # df_numeric = df.select_dtypes(include=[np.number, np.float64, np.int64])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f")
    # plt.show()
    # Previewing correlation shows little relevance between long / lat and median house value,
    # Though we might see better results with nonlinear regression.
    # However, we do see strong correlation with number of rooms and number of bedrooms. Insightful.
    
    # Starting regression model
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    # Scaling down longitude and latitude
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True) # batching
    
    model = HousingModel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    
    criterion = nn.MSELoss() # MSE Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    model.to(device)
    
    for epoch in range(100):
        epochLoss = 0.0
        for batch_X, batch_Y in dataLoader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
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
    # Calabasas, Bakersfield, Sacramento, Monterey
    test_lon = [-118.51390200, -119.08364300, -121.47644300, -123.80421100]
    test_lat = [34.20198200, 35.28472300, 38.53905900, 39.43978200]
    med_income = [8.4628, 7.9355, 8.3753, 5.2051] # in tens of thousands
    actual_prices = [700000, 575000, 419900, 399000]
    proximities = [0,1,0,2] # <1H Ocean, Inland, <1H Ocean, Near Ocean
    losses = []
    
    for i in range(4):
        price = predict_price(model, test_lat[i], test_lon[i], med_income[i], proximities[i], stats=norm_stats)
        print(f"Estimated price: ${price:,.0f}, actual price: ${actual_prices[i]:,.0f}")
        losses.append((price - actual_prices[i])**2)
    rmse = np.sqrt(np.mean(losses))
    print(f"RMSE on test locations: ${rmse:,.0f}")
        


    

if __name__ == "__main__":
    main()