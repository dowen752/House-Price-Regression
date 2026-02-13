import matplotlib.pyplot as plt
import pandas as pd
import folium
import os
from folium.plugins import HeatMap





def create_coord_heatmap(df):
    coords = df[["Latitude", "Longitude"]].dropna().values.tolist()

    m = folium.Map(location=[38.7946, 106.5348], zoom_start=5)

    HeatMap(coords, radius=8).add_to(m)
    map_dir = os.path.join("util", "maps", "housing_heatmap.html")

    m.save(map_dir)


def plot_housing_locations(df):
    plt.figure(figsize=(8,6))
    plt.scatter(df["Longitude"], df["Latitude"], s=1, alpha=0.5, c="red")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Housing Locations")
    plt.show()
    
    
    
def main():
    df = pd.read_csv("data/cleaned_df.csv")
    create_coord_heatmap(df)
    
if __name__ == "__main__":
    main()