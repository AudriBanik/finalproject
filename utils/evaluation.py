from sklearn.metrics import mean_squared_error
import torch
import numpy as np

def evaluate_rmse(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            output = model(X_batch)
            target = Y_batch.mean(dim=1)
            if output.dim() == 3:
                output = output.mean(dim=1)
            y_true.extend(target.cpu().numpy().flatten())
            y_pred.extend(output.cpu().numpy().flatten())
    return np.sqrt(mean_squared_error(y_true, y_pred))
