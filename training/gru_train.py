import torch
import torch.nn as nn
import torch.optim as optim
from utils.evaluation import evaluate_rmse

def train_gru(model, train_loader, test_loader, epochs=10, device="cpu"):
    """Train a GRU model in a centralized setup."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    rmse_values = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            target = Y_batch.mean(dim=1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        rmse = evaluate_rmse(model, test_loader)
        rmse_values.append(rmse)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}")

    return rmse_values
