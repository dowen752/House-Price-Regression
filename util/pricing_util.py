import pandas as pd
import numpy as np
import torch
from torch import nn
import os
from haversine import haversine, Unit


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
    floorplan_density = bedrooms / (area + eps)
    bb_ratio = bathrooms / (bedrooms + eps)
    land_use = area / (lot_area + eps)
    metro_dist = df["Dist_to_Metro"].values
    
    df["ListedPrice"] = np.log1p(df["ListedPrice"])
    price = df["ListedPrice"].values.reshape(-1, 1)
    
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
    metro_dist_mean, metro_dist_std = metro_dist.mean(), metro_dist.std()
    
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
    metro_dist = (metro_dist - metro_dist_mean) / metro_dist_std
    
    # Combine features into a single array    
    X = np.column_stack((lot_area, longitude, latitude, bedrooms, bathrooms, area, bb_ratio, metro_dist)) # Excluding land_use, floorplan_density
    
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


def save_model(model, path):
    torch.save(model.state_dict(), os.path.abspath(os.path.join("models", path)))

def load_model(model, path, device):
    model.load_state_dict(torch.load(os.path.abspath(os.path.join("models", path)), map_location = device))
    model.to(device)
    model.eval()
    return model

def house_dist_to_metro():
    metro_df = load_data("metro_locations.csv")
    metro_locations = metro_df[['latitude', 'longitude']].values
    housing_df = load_data("cleaned_df.csv")
    housing_locations = housing_df[['Latitude', 'Longitude']].values
    
    min_distances = []
    # Calculating haversine distance (spherical distance)
    for house in housing_locations:
        min_distance = float('inf')
        for metro in metro_locations:
            dist = haversine(house, metro, unit = Unit.MILES)
            if dist < min_distance:
                min_distance = dist
        # Storing min distance for each house
        min_distances.append(min_distance)
        
    housing_df['Dist_to_Metro'] = min_distances
    # Updatind cleaned_df.csv with new column
    housing_df.to_csv(os.path.abspath(os.path.join("data", "cleaned_df.csv")))
    
def main():
    house_dist_to_metro()
    print("Success")
    
# if __name__ == "__main__":
#     main()
    
    
    # Testing different feature combinations
    
    #X = np.column_stack((latitude, longitude))
    
    # Lat / Long Only:
    # Epoch 200, Loss: 0.6858661805269539
    # Validation RMSE (normalized): 0.8672901470846963
    # Validation RMSE (real): 0.6025274991989136
    
    # ////////////////////////////////////////////////////////////
    
    #X = np.column_stack((metro_dist))
    #X = X.reshape(-1, 1)
    
    # Metro Dist only:
    #Epoch 200, Loss: 0.9620908610122179
    #Validation RMSE (normalized): 0.993898403434803
    #Validation RMSE (real): 0.690485417842865
    
    # ////////////////////////////////////////////////////////////
    
    # Lat / Long / Metro Dist:
    #X = np.column_stack((latitude, longitude, metro_dist))
    
    # Metro / Lat / Long:
    # Epoch 200, Loss: 0.6440550195834329
    # Validation RMSE (normalized): 0.8298777618936833
    # Validation RMSE (real): 0.576536238193512
    
    # All features performing lower than before metro dist was added, but 
    # the combination of lat/long/metro is perfomring better than one or the other alone.
    # Need to check more combinations for optimization.