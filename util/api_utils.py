import os
import pandas as pd
import numpy as np
import json
import requests
import time

# Keeping for reference, but deprecated; dataset not sufficient
class rentcast_utils:
    
    # API_KEY = ""
       
    # Fetching listings for testing using rentcast api
    def fetch_listings():
        BASE_URL = "https://api.rentcast.io/v1/properties"

        HEADERS = {
            "X-Api-Key": API_KEY,
            "Accept": "application/json"
        }

        CITIES = [
            ("Los Angeles", "CA"),
            ("Santa Barbara", "CA"),
            ("San Jose", "CA"),
            ("San Diego", "CA"),
            ("Sacramento", "CA"),
            ("Portland", "OR"),
            ("Phoenix", "AZ"),
            ("Albuquerque", "NM"),
            ("Denver", "CO"),
            ("Las Vegas", "NV"),
            ("Austin", "TX"),
            ("Houston", "TX"),
            ("Atlanta", "GA")
        ]

        ALL_LISTINGS = []

        for city, state in CITIES:
            print(f"Fetching: {city}, {state}")

            params = {
                "city": city,
                "state": state,
                "limit": 50
            }
            
            try:
                response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10)
                response.raise_for_status()

                listings = response.json()
                ALL_LISTINGS.extend(listings)

                time.sleep(0.8)  # prevent hitting rate limits

            except requests.RequestException as e:
                print(f"Failed for {city}: {e}")
        json_path = os.path.join("data", "rentcast_multi_city.json")
        with open(json_path, "w") as f:
            json.dump(ALL_LISTINGS, f, indent=2)
            
        print(f"Saved {len(ALL_LISTINGS)} listings.")

        
    # Converting rentcast formatting to match rest of program
    def normalize_rentcast_df(df):
        df = df.rename(columns={
            "zipCode": "Zipcode",
            "longitude": "Longitude",
            "latitude": "Latitude",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "squareFootage": "Area",
            "ListedPrice": "Price"
        })
        
        # Keep only necessary columns
        return df[["Zipcode", "Longitude", "Latitude", "Bedrooms", "Bathrooms", "Area", "Price"]]


    # Pulling testing json data into dataframe
    def load_rentcast_json(file_name):
        full_path = os.path.abspath(os.path.join("data", file_name))
        with open(full_path) as f:
            listings = json.load(f)
        return pd.DataFrame(listings)
