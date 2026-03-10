import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LyapunovNet(nn.Module):
    def __init__(self, state_dim = 2, hidden_dim = 128):
        super().__init__()

        self.input_compress = nn.Linear(state_dim, 128)

        self.net = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        x = self.input_compress(x)
        V = self.net(x)
        # V= torch.relu(V) + 1e-6
        V = V**2 + 1e-6
        return V


