import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, num_nodes, hidden_size, layers=1):
        super().__init__()
        self.gru = nn.GRU(num_nodes, hidden_size, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])
