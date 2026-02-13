import pandas as pd
import numpy as np
import torch
from torch import nn
import os
from haversine import haversine, Unit


DF_PRICE_MEAN = 532439.9113795687
DF_PRICE_STD = 1574887.0879233533


# Main training loop
def train_model(model, dataloader, criterion, optimizer, device, val_dataset, stats):
    price_mean, price_std = stats["price_mean"], stats["price_std"]
    best_rmse = float("inf")

    checkpoint_path = "models/house_model.pth"

    if os.path.exists(checkpoint_path):
        cpkt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        best_rmse = cpkt.get("rmse", float("inf"))
        print(f"Loaded previous best RMSE: {best_rmse:,.0f}")
    
    for epoch in range(300):
        print(f"Epoch {epoch+1}")
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
    
        rmse, r2 = model_eval(model, val_dataset, device, stats)
        
        if rmse < best_rmse:
            best_rmse = rmse
            print(f"Saving new best model with RMSE: ${rmse:.2f} and R^2: {r2:.3f}")
            torch.save({
                "model_state": model.state_dict(),
                "x_mean": stats["x_mean"],
                "x_std": stats["x_std"],
                "y_mean": stats["price_mean"],
                "y_std": stats["price_std"],
                "feature_names": stats["feature_names"],
                "rmse": rmse,
                "r2": r2
            }, "models/house_model.pth")
            

# Loading data into dataframe
def load_data(file_name):
    full_path = os.path.abspath(os.path.join("data", file_name))
    data = pd.read_csv(full_path)
    return data


# Creating normalized, scaled data given dict of input data
def preprocess_training_data(df: pd.DataFrame, use_df: bool = False, test_price: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, float, float, dict]:
    # Extracting from df
    df.dropna(inplace=True)
    eps = 1e-8

    lot_area = df["LotArea"].values
    longitude = df["Longitude"].values
    latitude = df["Latitude"].values
    bedrooms = df["Bedroom"].values
    bathrooms = df["Bathroom"].values
    area = df["Area"].values
    bb_ratio = bathrooms / (bedrooms + eps)
    metro_dist = df["Dist_to_Metro"].values
    # floorplan_density = bedrooms / (area + eps)
    # land_use = area / (lot_area + eps)
    
    # Means and stds
    lot_area_mean, lot_area_std = lot_area.mean(), lot_area.std()
    longitude_mean, longitude_std = longitude.mean(), longitude.std()
    latitude_mean, latitude_std = latitude.mean(), latitude.std()
    bedrooms_mean, bedrooms_std = bedrooms.mean(), bedrooms.std()
    bathrooms_mean, bathrooms_std = bathrooms.mean(), bathrooms.std()
    area_mean, area_std = area.mean(), area.std()
    bb_ratio_mean, bb_ratio_std = bb_ratio.mean(), bb_ratio.std()
    metro_dist_mean, metro_dist_std = metro_dist.mean(), metro_dist.std()
    # floorplan_density_mean, floorplan_density_std = floorplan_density.mean(), floorplan_density.std()
    # land_use_mean, land_use_std = land_use.mean(), land_use.std() 
    
    # Standardizing
    lot_area = (lot_area - lot_area_mean) / lot_area_std
    longitude = (longitude - longitude_mean) / longitude_std
    latitude = (latitude - latitude_mean) / latitude_std
    bedrooms = (bedrooms - bedrooms_mean) / bedrooms_std
    bathrooms = (bathrooms - bathrooms_mean) / bathrooms_std
    area = (area - area_mean) / area_std
    bb_ratio = (bb_ratio - bb_ratio_mean) / bb_ratio_std
    metro_dist = (metro_dist - metro_dist_mean) / metro_dist_std
    #floorplan_density = (floorplan_density - floorplan_density_mean) / floorplan_density_std
    # land_use = (land_use - land_use_mean) / land_use_std
    
    df["ListedPrice"] = np.log1p(df["ListedPrice"])
    price = df["ListedPrice"].values.reshape(-1, 1)
    price_mean, price_std = price.mean(), price.std()
    price = (price - price_mean) / price_std
        
    # Combine features into a single array    
    X = np.column_stack((lot_area, latitude, longitude, bedrooms, bathrooms, area, bb_ratio, metro_dist)) # Excluding land_use, floorplan_density
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(price, dtype=torch.float32)
    
    # Storing stats for later use (e.g., for inference)
    stats = {
        "feature_names": ["LotArea", "Latitude", 
                          "Longitude", "Bedrooms", 
                          "Bathrooms", "Area", 
                          "BB_Ratio", "Dist_to_Metro"],
        
        "x_mean": [lot_area_mean, latitude_mean,
                    longitude_mean, bedrooms_mean, 
                    bathrooms_mean, area_mean, 
                    bb_ratio_mean, metro_dist_mean],
        
        "x_std": [lot_area_std, latitude_std, 
                   longitude_std, bedrooms_std, 
                   bathrooms_std, area_std, 
                   bb_ratio_std, metro_dist_std],
        
        "price_mean": price_mean,
        "price_std": price_std
        }
    
    
    return X_tensor, Y_tensor, stats


# Evaluation after training
def model_eval(model, val_loader, device, stats):
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
    
    # Converting to log1p space
    preds_log = preds * stats["price_std"] + stats["price_mean"]
    actuals_log = actuals * stats["price_std"] + stats["price_mean"]

    # R^2
    r2 = 1 - torch.sum((preds_log - actuals_log) ** 2) / torch.sum((actuals_log - actuals_log.mean()) ** 2)

    # Convert to dollars for RMSE
    rmse_dollars = torch.sqrt(torch.mean((torch.expm1(preds_log) - torch.expm1(actuals_log)) ** 2))
    
    return rmse_dollars.item(), r2.item()
    


def save_model(model, path):
    torch.save(model.state_dict(), os.path.abspath(os.path.join("models", path)))


def load_model(model, device, features, path):
    cpkt = torch.load(os.path.abspath(os.path.join("models", path)), map_location=device, weights_only=False)
    model.load_state_dict(cpkt["model_state"])
    model.to(device)
    
    # Standardization
    eps = 1e-8
    x_mean = cpkt["x_mean"]
    x_std = cpkt["x_std"]
    y_mean = cpkt["y_mean"]
    y_std = cpkt["y_std"]
    
    features.append(features[4] / (features[3] + eps)) # bb ratio, bathrooms / bedrooms
    features = add_metro_dist(features)    
    # Convert square feet to acres
    features[0] = features[0] / 43560

    X = np.array(features, dtype=np.float32).reshape(1, -1) # shape (1, n)

    # standardize w training stats
    x_mean = np.array(cpkt["x_mean"], dtype=np.float32).reshape(1, -1)
    x_std  = np.array(cpkt["x_std"],  dtype=np.float32).reshape(1, -1)
    
    Xs = (X - x_mean) / x_std
    X_tensor = torch.tensor(Xs, dtype=torch.float32, device=device)

    # y stats for converting back after prediction
    y_mean = float(cpkt["y_mean"])
    y_std  = float(cpkt["y_std"])

    return model, X_tensor, (y_mean, y_std)
        
        
    
    

# For adding distance to the cleaned_df.csv
def add_metro_dist_df():
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
    
    
# For adding distance to a single row of features
def add_metro_dist(features: list) -> list:
    metro_df = load_data("metro_locations.csv")
    metro_locations = metro_df[['latitude', 'longitude']].values
    house_location = (features[1], features[2])  # Latitude, Longitude
    min_distance = float('inf')
    for metro in metro_locations:
        dist = haversine(house_location, metro, unit = Unit.MILES)
        if dist < min_distance:
            min_distance = dist
    features.append(min_distance)    
        
    return features
    
    
def main():
    pass
    
# if __name__ == "__main__":
#     main()




# # Extracting data from csv
# def extracting_data(df):
#     # df.head()
#     # df.info()
#     # df.describe()
#     df.dropna(inplace=True)
#     # Need to extract and normalize for:
#     # zipcode, Longitude, latitude, bedroom, room, and price
#     zipcode = df['Zipcode'].values
#     longitude = df['Longitude'].values
#     latitude = df['Latitude'].values
#     bedrooms = df['Bedroom'].values
#     bathrooms = df['Bathroom'].values
#     area = df['Area'].values
#     print(df[""])
#     df["Price"] = np.log1p(df["Price"])
#     price = df['Price']
#     data = pd.DataFrame({"Zipcode" : zipcode, 
#                         "Longitude" : longitude, 
#                         "Latitude" : latitude, 
#                         "Bedrooms" : bedrooms, 
#                         "Bathrooms" : bathrooms, 
#                         "Area" : area, 
#                         "Price": price
#                         })
#     return data