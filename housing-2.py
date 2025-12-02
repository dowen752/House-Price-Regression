import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

def load_data(file_name):
    full_path = os.path.abspath(os.path.join("data", file_name))
    data = pd.read_csv(full_path)
    return data

def preprocess_data(df):
    # df.head()
    # df.info()
    # df.describe()
    df.dropna(inplace=True)
    # Need to extract and normalize for:
    # zipcode, Longitude, latitude, bedroom, room, and price
    zipcode = df['Zipcode'].values
    longitude = df['Longitude'].values
    latitude = df['Latitude'].values
    bedrooms = df['Bedroom'].values
    bathrooms = df['Bathroom'].values
    area = df['Area'].values
    df["Price"] = np.log1p(df["Price"])
    price = df['Price'].values.reshape(-1, 1)
    
    # Normalizing
    zipcode_mean, zipcode_std = zipcode.mean(), zipcode.std()
    longitude_mean, longitude_std = longitude.mean(), longitude.std()
    latitude_mean, latitude_std = latitude.mean(), latitude.std()
    bedrooms_mean, bedrooms_std = bedrooms.mean(), bedrooms.std()
    bathrooms_mean, bathrooms_std = bathrooms.mean(), bathrooms.std()
    area_mean, area_std = area.mean(), area.std()
    price_mean, price_std = price.mean(), price.std()
    
    print(f"Price mean: {price_mean}, std: {price_std}")
    
    # Scaling to 0 - 1
    zipcode = (zipcode - zipcode_mean) / zipcode_std
    longitude = (longitude - longitude_mean) / longitude_std
    latitude = (latitude - latitude_mean) / latitude_std
    bedrooms = (bedrooms - bedrooms_mean) / bedrooms_std
    bathrooms = (bathrooms - bathrooms_mean) / bathrooms_std
    price = (price - price_mean) / price_std
    area = (area - area_mean) / area_std
    # Combine features into a single array
    X = np.column_stack((zipcode, longitude, latitude, bedrooms, bathrooms, area))
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(price, dtype=torch.float32)
    
    return X_tensor, Y_tensor, price_mean, price_std
    
class HousingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.net(x)
    
def main():
    data = load_data("original_extracted_df.csv")
    X, Y, price_mean, price_std = preprocess_data(data)
    
    # numeric_data = data.select_dtypes(include=[np.number, np.float64, np.int64])
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    # plt.show()
    
    # Outside of the actual price estimate, the strongest coorrelations are 
    # clearly with area, bedrooms, and bathrooms. I should definitely use these
    # features in my model.
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
    model = HousingModel(input_dim=X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(100):
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
        
    model.eval()
    
    # Prediction example
    zipcode = 90230
    longitude = -118.39131800
    latitude = 34.00319300
    bedrooms = 3
    bathrooms = 2
    area = 1289.0
    # Normalize input
    zipcode_norm = (zipcode - data['Zipcode'].mean()) / data['Zipcode'].std()
    longitude_norm = (longitude - data['Longitude'].mean()) / data['Longitude'].std()
    latitude_norm = (latitude - data['Latitude'].mean()) / data['Latitude'].std()
    bedrooms_norm = (bedrooms - data['Bedroom'].mean()) / data['Bedroom'].std()
    bathrooms_norm = (bathrooms - data['Bathroom'].mean()) / data['Bathroom'].std()
    area_norm = (area - data['Area'].mean()) / data['Area'].std()
    
    input = torch.tensor([[zipcode_norm, longitude_norm, latitude_norm, bedrooms_norm, bathrooms_norm, area_norm]], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = model(input)
    # Un-normalize output
    pred_price = pred_norm.item() * data['Price'].std() + data['Price'].mean()
    pred_price = np.expm1(pred_norm.item() * price_std + price_mean)
    print(f"Estimated price: {pred_price:,.2f}, actual price: 859,000.0")

    
if __name__ == "__main__":
    main()