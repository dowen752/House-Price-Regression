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
    data = pu.load_data("cleaned_df.csv")
    X, Y, price_mean, price_std = pu.preprocess_data(data)
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
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    for epoch in range(200):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            # Forward pass
            predictions = model(batch_X)
            # Loss
            loss = criterion(predictions, batch_Y)
            # Zero gradients
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}")

    pu.model_eval(model, val_dataset, device, price_mean, price_std)


    
if __name__ == "__main__":
    main()