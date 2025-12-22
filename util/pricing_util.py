import pandas as pd
import numpy as np
import torch
from torch import nn
import os


# Loading data into dataframe
def load_data(file_name):
    full_path = os.path.abspath(os.path.join("data", file_name))
    data = pd.read_csv(full_path)
    return data


# Creating normalized, scaled data given dict of input data
def preprocess_data(df: pd.DataFrame):
    # Extracting from df
    df.dropna(inplace=True)
    eps = 1e-8

    lot_area = df["LotArea"].values
    longitude = df["Longitude"].values
    latitude = df["Latitude"].values
    bedrooms = df["Bedroom"].values
    bathrooms = df["Bathroom"].values
    area = df["Area"].values
    df["ListedPrice"] = np.log1p(df["ListedPrice"])
    price = df["ListedPrice"].values.reshape(-1, 1)
    floorplan_density = bedrooms / (area + eps)
    bb_ratio = bathrooms / (bedrooms + eps)
    land_use = area / (lot_area + eps)
    
    # Normalizing
    lot_area_mean, lot_area_std = lot_area.mean(), lot_area.std()
    longitude_mean, longitude_std = longitude.mean(), longitude.std()
    latitude_mean, latitude_std = latitude.mean(), latitude.std()
    bedrooms_mean, bedrooms_std = bedrooms.mean(), bedrooms.std()
    bathrooms_mean, bathrooms_std = bathrooms.mean(), bathrooms.std()
    area_mean, area_std = area.mean(), area.std()
    price_mean, price_std = price.mean(), price.std()
    floorplan_density_mean, floorplan_density_std = floorplan_density.mean(), floorplan_density.std()
    bb_ratio_mean, bb_ratio_std = bb_ratio.mean(), bb_ratio.std()
    land_use_mean, land_use_std = land_use.mean(), land_use.std() 
    
    # Scaling to 0 - 1
    lot_area = (lot_area - lot_area_mean) / lot_area_std
    longitude = (longitude - longitude_mean) / longitude_std
    latitude = (latitude - latitude_mean) / latitude_std
    bedrooms = (bedrooms - bedrooms_mean) / bedrooms_std
    bathrooms = (bathrooms - bathrooms_mean) / bathrooms_std
    price = (price - price_mean) / price_std
    area = (area - area_mean) / area_std
    floorplan_density = (floorplan_density - floorplan_density_mean) / floorplan_density_std
    bb_ratio = (bb_ratio - bb_ratio_mean) / bb_ratio_std
    land_use = (land_use - land_use_mean) / land_use_std
    
    # Combine features into a single array
    X = np.column_stack((lot_area, longitude, latitude, bedrooms, bathrooms, area, floorplan_density, bb_ratio, land_use))
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(price, dtype=torch.float32)
    
    return X_tensor, Y_tensor, price_mean, price_std

# Evaluation after training
def model_eval(model, val_loader, device, price_mean, price_std):
    model.eval()
    
    preds = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            outputs = model(batch_X)
            preds.append(outputs)
            actuals.append(batch_Y)
    
    preds = torch.cat(preds)
    actuals = torch.cat(actuals)
    
    mse_norm = nn.MSELoss()(preds, actuals).item()
    rmse_norm = np.sqrt(mse_norm)
    
    preds_real = preds * price_std + price_mean
    actuals_real = actuals * price_std + price_mean
    rmse_real = nn.MSELoss()(preds_real, actuals_real).sqrt().item()
    
    print(f"Validation RMSE (normalized): {rmse_norm}")
    print(f"Validation RMSE (real): {rmse_real}")

# Extracting data from csv
def extracting_data(df):
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
    print(df[""])
    df["Price"] = np.log1p(df["Price"])
    price = df['Price']
    data = pd.DataFrame({"Zipcode" : zipcode, 
                        "Longitude" : longitude, 
                        "Latitude" : latitude, 
                        "Bedrooms" : bedrooms, 
                        "Bathrooms" : bathrooms, 
                        "Area" : area, 
                        "Price": price
                        })
    return data