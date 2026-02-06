"""
Inspect embedding files content
"""
import numpy as np
import pickle
import os
import sys

BASE_DIR = r"d:\Final_Year_Project\ShopWhatYouSee"
AGMAN_DIR = os.path.join(BASE_DIR, "agman_output")
PKL_PATH = os.path.join(BASE_DIR, "backend", "data", "embeddings", "product_embeddings.pkl")

def inspect():
    # 1. Inspect labels.npy
    try:
        labels_path = os.path.join(AGMAN_DIR, "labels.npy")
        if os.path.exists(labels_path):
            print(f"--- {labels_path} ---")
            labels = np.load(labels_path, allow_pickle=True)
            print(f"Shape: {labels.shape}")
            print(f"First 10 labels: {labels[:10]}")
            print(f"Type of first label: {type(labels[0])}")
        else:
            print(f"--- {labels_path} NOT FOUND ---")
    except Exception as e:
        print(f"Error reading labels: {e}")

    # 2. Inspect product_embeddings.pkl
    try:
        if os.path.exists(PKL_PATH):
            print(f"\n--- {PKL_PATH} ---")
            with open(PKL_PATH, 'rb') as f:
                data = pickle.load(f)
            print(f"Type: {type(data)}")
            if isinstance(data, dict):
                print(f"Keys count: {len(data)}")
                first_key = list(data.keys())[0]
                first_val = data[first_key]
                print(f"First Key: {first_key}")
                if hasattr(first_val, 'shape'):
                    print(f"First Value Shape: {first_val.shape}")
                elif isinstance(first_val, list):
                    print(f"First Value Length: {len(first_val)}")
            else:
                print("Data is not a dict")
        else:
            print(f"--- {PKL_PATH} NOT FOUND ---")
    except Exception as e:
        print(f"Error reading pickle: {e}")

if __name__ == "__main__":
    inspect()
