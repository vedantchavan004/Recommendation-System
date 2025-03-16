import pandas as pd
import torch
import os
import zipfile
import urllib.request

DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "data"

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Dataset extracted.")

def load_ratings():
    # Path to the u.data file
    ratings_file = os.path.join(DATA_DIR, "ml-100k", "u.data")
    ratings = pd.read_csv(ratings_file, sep='\t', header=None, 
                          names=['user', 'item', 'rating', 'timestamp'])
    # Treat ratings >= 4 as positive feedback
    ratings = ratings[ratings['rating'] >= 4]
    # Adjust to zero-indexed
    ratings['user'] = ratings['user'] - 1
    ratings['item'] = ratings['item'] - 1
    
    # Use maximum value + 1 to compute number of users and items
    num_users = int(ratings['user'].max()) + 1
    num_items = int(ratings['item'].max()) + 1
    
    return ratings, num_users, num_items

if __name__ == '__main__':
    download_and_extract()
    ratings, num_users, num_items = load_ratings()
    print(f"Users: {num_users}, Items: {num_items}")
    print(ratings.head())
