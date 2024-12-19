import os
import requests

def download_dataset():
    datasets = {
        "PEMS-BAY.csv": "https://zenodo.org/records/5146275/files/PEMS-BAY.csv?download=1",
        "METR-LA.csv": "https://zenodo.org/records/5146275/files/METR-LA.csv?download=1",
    }
    os.makedirs("data", exist_ok=True)
    for filename, url in datasets.items():
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename} to {filepath}")
