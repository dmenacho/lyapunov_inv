import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

class LyapunovNet(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        V = self.net(x)
        V=torch.relu(V) + 1e-6
        return V

def zubov_loss(x, V_net, f_torch, device, mu=0.1, transform="exp"):
    # https://git.uwaterloo.ca/hybrid-systems-lab/lyznet
    # adapterted from lyznet/src/lyznet/neural_learner.py
    x.requires_grad = True
    V = V_net(x).squeeze()
    V_grad = torch.autograd.grad( outputs=V.sum(), inputs=x, create_graph=True, retain_graph=True )[0]
    f= f_torch(x)
    V_dot = (V_grad*f).sum(dim=1) 
    norm_sq= (x**2).sum(dim=1) 
    if transform == "exp":
        zubov_residual = V_dot + mu*norm_sq*(1 - V)
    else: 
        zubov_residual = V_dot + mu*norm_sq*(1 - V)*(1 + V)
    pde_loss = zubov_residual**2
    
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero_tensor.requires_grad_(True)
    V_zero = V_net(zero_tensor)
    V_grad_zero = torch.autograd.grad(outputs=V_zero.sum(), inputs=zero_tensor, create_graph=True)[0]
    
    orign_loss =  (V_grad_zero**2).sum() + (V_zero**2)
    loss = (pde_loss + orign_loss).mean()
    
    return loss

def prepare_trajectory_data(trajectory_dict, dt=0.1):
    trajectory = trajectory_dict['trajectory']
    dx_dt = np.diff(trajectory, axis=0) / dt
    dx_dt = np.vstack([dx_dt[0], dx_dt])
    return trajectory.astype(np.float32), dx_dt.astype(np.float32)
    
class LyapunovLearner:    
    def __init__(self, state_dim=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lyapunov = LyapunovNet(state_dim=state_dim).to(device)
        self.optimizer = optim.Adam(self.lyapunov.parameters(), lr=1e-3)
    
    def load_dataset(self, dataset_path, train_split=0.8):

        with open(dataset_path, 'rb') as f:
            full_dataset = pickle.load(f)
        
        stable_data = [d for d in full_dataset if not d['stable']]
        print(f"{len(stable_data)} stable trajectories")
        
        n_train = int(len(stable_data) * train_split)
        train_data = stable_data[:n_train]
        test_data = stable_data[n_train:]
        
        return train_data, test_data
    
    def prepare_collocation_points(self, train_data, num_points=50000):
        all_x = []
        all_dx_dt = []
        for traj_data in train_data:
            x, dx_dt = prepare_trajectory_data(traj_data)
            all_x.append(x)
            all_dx_dt.append(dx_dt)
        
        all_x = np.vstack(all_x)
        all_dx_dt = np.vstack(all_dx_dt)
    
        if len(all_x) > num_points:
            idx = np.random.choice(len(all_x), num_points, replace=False)
            all_x = all_x[idx]
            all_dx_dt = all_dx_dt[idx]
        
        x_tensor = torch.tensor(all_x, dtype=torch.float32).to(self.device)
        dx_dt_tensor = torch.tensor(all_dx_dt, dtype=torch.float32).to(self.device)
        
        return x_tensor, dx_dt_tensor
    
    def train(self, x_train, dx_dt_train, epochs=10, batch_size=256, mu=0.1, transform="exp"):
        n_samples = x_train.shape[0]
        
        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            x_shuffled = x_train[perm]
            f_shuffled = dx_dt_train[perm]
            
            epoch_loss = 0
            n_batches = 0
            
            pbar = tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch + 1}", leave=False)
            for i in pbar:
                x_batch = x_shuffled[i:i+batch_size]
                f_batch = f_shuffled[i:i+batch_size]
                
                loss = zubov_loss(x_batch, self.lyapunov, lambda x: f_batch[:x.shape[0]],  self.device, mu=mu, transform=transform)
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=loss.item())
            
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
            
if __name__ == "__main__":
    learner = LyapunovLearner(state_dim=8)
    train_data, test_data = learner.load_dataset('data/training_data_y_up.pkl')
    
    x_train, f_train = learner.prepare_collocation_points(train_data, num_points=100000)
    learner.train(x_train, f_train, epochs=25, batch_size=256, mu=0.1, transform="exp")
    Path('models').mkdir(exist_ok=True)
    torch.save(learner.lyapunov.state_dict(), 'models/lyapunov_zubov_lya.pt')