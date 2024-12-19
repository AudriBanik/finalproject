import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

def preprocess_csv(filepath, seq_length=12, pred_length=12, sample_frac=1.0):
    """
    Preprocess the traffic data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        seq_length (int): Sequence length for the input data.
        pred_length (int): Prediction length for the output data.
        sample_frac (float): Fraction of data to sample (default is 1.0, i.e., all data).
    
    Returns:
        Tuple[TensorDataset, TensorDataset, TensorDataset]: Training, validation, and test datasets.
    """
    traffic_data = pd.read_csv(filepath)
    traffic_data = traffic_data.select_dtypes(include=[np.number]).values

    # Sample fraction of the data if specified
    if sample_frac < 1.0:
        num_samples = int(len(traffic_data) * sample_frac)
        traffic_data = traffic_data[:num_samples]

    # Normalize data
    mean = traffic_data.mean()
    std = traffic_data.std()
    traffic_data = (traffic_data - mean) / std

    # Split data into training, validation, and test sets
    num_samples = traffic_data.shape[0]
    num_train = int(num_samples * 0.6)
    num_val = int(num_samples * 0.2)
    train_data = traffic_data[:num_train]
    val_data = traffic_data[num_train:num_train + num_val]
    test_data = traffic_data[num_train + num_val:]

    def create_sequences(data):
        """
        Create sequences for supervised learning.
        """
        X, Y = [], []
        for i in range(len(data) - seq_length - pred_length):
            X.append(data[i:i + seq_length])
            Y.append(data[i + seq_length:i + seq_length + pred_length])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_sequences(train_data)
    X_val, Y_val = create_sequences(val_data)
    X_test, Y_test = create_sequences(test_data)

    # Convert to PyTorch TensorDataset
    return (
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(Y_train, dtype=torch.float32)),
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(Y_val, dtype=torch.float32)),
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(Y_test, dtype=torch.float32)),
    )

def compute_adjacency_from_csv(csv_file, threshold=0.5):
    """
    Compute an adjacency matrix from a CSV file using correlation.
    
    Args:
        csv_file (str): Path to the CSV file.
        threshold (float): Correlation threshold for adjacency (default is 0.5).
    
    Returns:
        numpy.ndarray: Adjacency matrix.
    """
    data = pd.read_csv(csv_file).select_dtypes(include=[np.number])
    correlation_matrix = data.corr().abs()  # Compute absolute correlation
    adjacency_matrix = (correlation_matrix >= threshold).astype(int).values

    # Ensure no self-loops
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix
