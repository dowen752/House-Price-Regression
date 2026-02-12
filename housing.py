import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
import os
import requests
import time
import json
from util import pricing_util as pu
    
    
# NN Class
class HousingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)
    
def main():
    choice = input("Train or run model? (\"t\" / \"r\"): ").strip().lower()
    if choice == "t":
        data = pu.load_data("cleaned_df.csv")
        X, Y, stats = pu.preprocess_training_data(data)
        dataset = TensorDataset(X, Y)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
        model = HousingModel(input_dim=X.shape[1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Using device: {device}")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        pu.train_model(model, dataloader, criterion, optimizer, device, val_dataset, stats)
        
        
    elif  choice == "r":
        
        # Prompt for inputs
        features = [
            float(input("Enter Lot Area (sq ft): ")),
            float(input("Enter Latitude: ")),
            float(input("Enter Longitude: ")),
            float(input("Enter Number of Bedrooms: ")),
            float(input("Enter Number of Bathrooms: ")),
            float(input("Enter Interior Area (sq ft): "))
        ]
        
        model = HousingModel(input_dim=8)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, X, price_stats = pu.load_model(model, device, features, "house_model.pth")
        y_mean, y_std = price_stats
        
        use_actual = input("Do you want to enter an actual price for comparison? (y/n): ").strip().lower()
        if use_actual == 'y':
            actual_price = float(input("Enter Actual Price (for comparison): "))
        else:
            actual_price = 0.0
        
        model.eval()
        with torch.no_grad():
            X = X.to(next(model.parameters()).device)
            pred = model(X).cpu().item()
            pred_price = np.expm1(pred * y_std + y_mean)
            if use_actual == 'y':
                print(f"Predicted Price: ${pred_price:.2f}, Actual Price: ${actual_price:.2f} \n")
                print(f"Prediction is within {abs(pred_price - actual_price) / actual_price * 100:.2f}% of actual price")
            else:
                print(f"Predicted Price: ${pred_price:.2f}")
    else:
        print("Invalid choice. Please enter t or r.")

    
if __name__ == "__main__":
    main()