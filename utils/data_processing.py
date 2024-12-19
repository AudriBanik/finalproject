import pandas as pd
import torch
from torch.utils.data import TensorDataset

def preprocess_csv(filepath, seq_length=12, pred_length=12):
    traffic_data = pd.read_csv(filepath).select_dtypes(include=[float]).values
    mean, std = traffic_data.mean(), traffic_data.std()
    traffic_data = (traffic_data - mean) / std
    num_samples = traffic_data.shape[0]
    num_train = int(num_samples * 0.6)
    num_val = int(num_samples * 0.2)

    train_data = traffic_data[:num_train]
    val_data = traffic_data[num_train:num_train + num_val]
    test_data = traffic_data[num_train + num_val:]

    def create_sequences(data):
        X, Y = [], []
        for i in range(len(data) - seq_length - pred_length):
            X.append(data[i:i + seq_length])
            Y.append(data[i + seq_length:i + seq_length + pred_length])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    return (
        TensorDataset(*create_sequences(train_data)),
        TensorDataset(*create_sequences(val_data)),
        TensorDataset(*create_sequences(test_data)),
    )
