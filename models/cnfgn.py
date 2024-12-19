import torch
import torch.nn as nn

class CNFGNN(nn.Module):
    def __init__(self, adjacency_matrix, num_nodes, hidden_size=64):
        super().__init__()
        self.adjacency = nn.Parameter(torch.tensor(adjacency_matrix, dtype=torch.float32), requires_grad=False)
        self.fc1 = nn.Linear(num_nodes, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.einsum('ij,bjk->bik', self.adjacency, x)
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
