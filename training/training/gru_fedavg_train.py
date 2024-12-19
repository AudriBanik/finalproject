import torch
import torch.nn as nn
import torch.optim as optim
from utils.evaluation import evaluate_rmse
import numpy as np

def train_gru_fedavg(num_nodes, train_loader, test_loader, epochs=10, device="cpu", node_fraction=0.5):
    """Federated Averaging for GRU."""
    local_models = [GRUModel(1, hidden_size=100).to(device) for _ in range(num_nodes)]
    global_model = GRUModel(1, hidden_size=100).to(device)
    rmse_values = []

    for epoch in range(epochs):
        participating_nodes = np.random.choice(num_nodes, int(num_nodes * node_fraction), replace=False)
        local_states = []

        for node in participating_nodes:
            model = local_models[node]
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                target = Y_batch.mean(dim=1)
                loss = nn.MSELoss()(output, target)
                loss.backward()
                optimizer.step()
            local_states.append(model.state_dict())

        # Federated Averaging
        global_state = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        for state in local_states:
            for k, v in state.items():
                global_state[k] += v / len(participating_nodes)
        global_model.load_state_dict(global_state)

        # Push global weights to local models
        for node in participating_nodes:
            local_models[node].load_state_dict(global_model.state_dict())

        rmse = evaluate_rmse(global_model, test_loader)
        rmse_values.append(rmse)
        print(f"Epoch {epoch + 1}, Global RMSE: {rmse:.4f}")

    return rmse_values
