import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.gru import GRUModel, GRUWithGN, GRUWithFMTL, CNFGNN
from training.gru_train import train_gru
from training.gru_fedavg_train import train_gru_fedavg
from training.gru_fmtl_train import train_gru_fmtl
from training.cnfgnn_train import train_cnfgnn_federated
from utils.data_preprocessing import preprocess_csv, compute_adjacency_from_csv

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
datasets = ["PEMS-BAY", "METR-LA"]

# Experiment results
results = {}

def main():
    for dataset in datasets:
        print(f"\n--- Running experiments on {dataset} dataset ---")
        
        # Paths
        csv_file = f"data/{dataset}.csv"
        if not os.path.exists(csv_file):
            print(f"Dataset {csv_file} not found. Please ensure the dataset exists in the 'data' folder.")
            continue
        
        # Load adjacency matrix and dataset
        adjacency_matrix = compute_adjacency_from_csv(csv_file)
        num_nodes = adjacency_matrix.shape[0]
        train_data, val_data, test_data = preprocess_csv(csv_file)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # --- GRU (Centralized) ---
        print("\nTraining GRU (Centralized, 1 Layer)...")
        centralized_gru = GRUModel(num_nodes=num_nodes, hidden_size=100).to(device)
        results[f"{dataset}_GRU_Centralized"] = train_gru(centralized_gru, train_loader, test_loader, device=device)

        # --- GRU + GN (Centralized) ---
        print("\nTraining GRU + GN (Centralized)...")
        gru_with_gn = GRUWithGN(adjacency_matrix=adjacency_matrix, num_nodes=num_nodes, hidden_size=100).to(device)
        results[f"{dataset}_GRU_GN_Centralized"] = train_gru(gru_with_gn, train_loader, test_loader, device=device)

        # --- GRU + FedAvg ---
        print("\nTraining GRU + FedAvg...")
        fedavg_rmse = train_gru_fedavg(num_nodes, train_loader, test_loader, device=device)
        results[f"{dataset}_GRU_FedAvg"] = fedavg_rmse

        # --- GRU + FMTL ---
        print("\nTraining GRU + FMTL...")
        fmtl_rmse = train_gru_fmtl(num_nodes, train_loader, test_loader, adjacency_matrix, device=device)
        results[f"{dataset}_GRU_FMTL"] = fmtl_rmse

        # --- CNFGNN (Federated) ---
        print("\nTraining CNFGNN (Federated)...")
        cnfgnn_rmse = train_cnfgnn_federated(num_nodes, train_loader, test_loader, adjacency_matrix, device=device)
        results[f"{dataset}_CNFGNN"] = cnfgnn_rmse

    # Save results to a DataFrame and display
    results_df = pd.DataFrame(results)
    print("\n--- RMSE Comparison ---")
    print(results_df)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/rmse_comparison.csv", index=False)
    print("\nResults saved to 'results/rmse_comparison.csv'.")

if __name__ == "__main__":
    main()
