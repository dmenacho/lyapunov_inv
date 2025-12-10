import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ParametricSurrogate(nn.Module):

    def __init__(self, state_dim=8, param_dim=6, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + param_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x0, theta, t):
        x_in = torch.cat([x0, theta, t], dim=1)
        return self.net(x_in)
    

def train_parametric_surrogate(dataset_path, epochs=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    surrogate = ParametricSurrogate(state_dim=8, param_dim=6, hidden_dim=256).to(device)
    optimizer = optim.Adam(surrogate.parameters(), lr=1e-3)
    
    X0_list = []
    theta_list = []
    t_list = []
    X_list = []
    
    for traj_dict in dataset:
        trajectory = np.array(traj_dict['trajectory'])
        theta = np.array(traj_dict['theta'])
        x0 = trajectory[0]

        for i, t_val in enumerate(np.linspace(0, 15, len(trajectory))):
            X0_list.append(x0)
            theta_list.append(theta)
            t_list.append([t_val])
            X_list.append(trajectory[i])
    
    X0_tensor = torch.tensor(np.array(X0_list), dtype=torch.float32, device=device)
    theta_tensor = torch.tensor(np.array(theta_list), dtype=torch.float32, device=device)
    t_tensor = torch.tensor(np.array(t_list), dtype=torch.float32, device=device)
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32, device=device)

    batch_size = 512
    for epoch in range(epochs):
        epoch_loss = 0
        
        indices = torch.randperm(len(X_tensor))
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            x0_batch = X0_tensor[batch_idx]
            theta_batch = theta_tensor[batch_idx]
            t_batch = t_tensor[batch_idx]
            x_batch = X_tensor[batch_idx]

            x_pred = surrogate(x0_batch, theta_batch, t_batch)
            loss = torch.mean((x_pred - x_batch) ** 2)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surrogate.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / (len(X_tensor) // batch_size)
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6e}")

    Path('models').mkdir(exist_ok=True)
    torch.save(surrogate.state_dict(), 'models/parametric_surrogate.pt')
    return surrogate


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surrogate = train_parametric_surrogate( dataset_path='data/training_data_10k_01_b_negative.pkl', epochs=100)