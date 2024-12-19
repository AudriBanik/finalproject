import torch
import torch.nn as nn

class GRUWithFMTL(nn.Module):
    def __init__(self, num_nodes, hidden_size, adjacency_matrix, lambda_reg=0.1):
        super().__init__()
        self.gru = nn.GRU(num_nodes, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_nodes)
        self.adjacency = nn.Parameter(torch.tensor(adjacency_matrix, dtype=torch.float32), requires_grad=False)
        self.lambda_reg = lambda_reg

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

    def regularization_loss(self, weights):
        laplacian = torch.diag(self.adjacency.sum(dim=1)) - self.adjacency
        return self.lambda_reg * torch.trace(weights @ laplacian @ weights.T)
