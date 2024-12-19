import os
import requests

# URLs for datasets
DATASETS = {
    "PEMS-BAY.csv": "https://zenodo.org/records/5146275/files/PEMS-BAY.csv?download=1",
    "METR-LA.csv": "https://zenodo.org/records/5146275/files/METR-LA.csv?download=1"
}

DATA_DIR = "data"

def download_file(url, dest_path):
    """
    Download a file from a URL and save it to a destination path.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {dest_path}")
    else:
        print(f"Failed to download: {url} (Status code: {response.status_code})")

def main():
    """
    Download all datasets and save them to the `data/` directory.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    for file_name, url in DATASETS.items():
        dest_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(dest_path):
            print(f"Downloading {file_name}...")
            download_file(url, dest_path)
        else:
            print(f"{file_name} already exists. Skipping download.")

if __name__ == "__main__":
    main()
