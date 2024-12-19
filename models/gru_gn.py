import torch
import torch.nn as nn

class GRUWithGN(nn.Module):
    def __init__(self, adjacency_matrix, num_nodes, hidden_size=64):
        super().__init__()
        self.adjacency = nn.Parameter(torch.tensor(adjacency_matrix, dtype=torch.float32), requires_grad=False)
        self.gru = nn.GRU(num_nodes, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_nodes)
        self.fc2 = nn.Linear(num_nodes, 256)
        self.fc3 = nn.Linear(256, num_nodes)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n[-1]
        node_features = self.fc1(h_n)
        node_features = torch.einsum('ij,bj->bi', self.adjacency, node_features)
        h = torch.relu(self.fc2(node_features))
        return self.fc3(h)
