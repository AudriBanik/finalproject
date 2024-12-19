import os
from torch.utils.data import DataLoader
from models.gru import GRUModel
from models.gru_gn import GRUWithGN
from models.gru_fmtl import GRUWithFMTL
from models.cnfgn import CNFGNN
from utils.data_processing import preprocess_csv
from utils.download import download_dataset
from utils.evaluation import evaluate_rmse

download_dataset()

datasets = ["data/PEMS-BAY.csv", "data/METR-LA.csv"]
for dataset in datasets:
    train_data, val_data, test_data = preprocess_csv(dataset)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = GRUModel(num_nodes=325, hidden_size=100)
    print(f"Evaluating {dataset}")
    print(evaluate_rmse(model, test_loader))
