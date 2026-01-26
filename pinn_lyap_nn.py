import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt

class LyapunovNet(nn.Module):
    def __init__(self, state_dim = 2, hidden_dim = 128):
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

class NNDynamics:
    def __init__(self, input_dim = 1, hidden_dim = 8, output_dim=  1,device = 'cpu'):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.state_dim = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

        np.random.seed(42)
        self.X_train = np.random.randn(50, input_dim).astype(np.float32)
        self.y_train = np.sin(self.X_train).astype(np.float32) 
        self.X_train_tensor = torch.tensor(self.X_train, device=device, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, device=device, dtype=torch.float32)

    def flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def unflatten_params(self, flat_params, model):
        idx = 0
        for param in model.parameters():
            size = param.data.numel()
            param.data = torch.tensor(
                flat_params[idx:idx+size].reshape(param.shape),
                dtype=torch.float32,
                device=self.device
            )
            idx += size

    def simulate_trajectory( self, w_init, learning_rate, num_steps):
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim)
        ).to(self.device)
        
        self.unflatten_params(w_init, model)
        
        trajectory = [w_init.copy()]
        loss_fn = nn.MSELoss()
        
        for step in range(num_steps):
            y_pred = model(self.X_train_tensor)
            loss = loss_fn(y_pred, self.y_train_tensor)
            
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
            
            w_current = self.flatten_params(model)
            trajectory.append(w_current.copy())
        
        trajectory = np.array(trajectory)
        
        # Approximate velocities via finite differences
        velocities = np.diff(trajectory, axis=0) / (1.0 / learning_rate)  
        velocities = np.vstack([velocities[0], velocities])
        
        return trajectory, velocities
    
    def generate_collocation_points(self, num_trajectories= 100, num_steps = 10, learning_rate_range = (1e-3, 0.1)):
        all_states = []
        all_velocities = []
        
        for _ in tqdm(range(num_trajectories), desc="Generating trajectories"):
            w_init = np.random.randn(self.state_dim).astype(np.float32) * 0.1
            lr = np.random.uniform(*learning_rate_range)
            trajectory, velocities = self.simulate_trajectory(w_init, lr, num_steps)
            
            all_states.append(trajectory)
            all_velocities.append(velocities)
        
        x_array = np.vstack(all_states)
        f_array = np.vstack(all_velocities)
        
        x_mean = x_array.mean(axis=0)
        x_std = x_array.std(axis=0) + 1e-8
        x_array = (x_array - x_mean) / x_std

        x_tensor = torch.tensor(x_array, dtype=torch.float32, device=self.device)
        f_tensor = torch.tensor(f_array, dtype=torch.float32, device=self.device)
        
        return x_tensor, f_tensor
    
class LyapunovLearner:
    def __init__(self, state_dim = 2, device= 'cpu'):
        self.device = device
        self.state_dim = state_dim
        self.lyapunov = LyapunovNet(state_dim=state_dim, hidden_dim=128).to(device)
        self.optimizer = optim.Adam(self.lyapunov.parameters(), lr=1e-3)
    
    def train(self, x_train, f_train, epochs=10, batch_size=256, mu=0.1, transform="exp"):
        n_samples = x_train.shape[0]
        losses = []
        
        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            x_shuffled = x_train[perm]
            f_shuffled = f_train[perm]
            
            epoch_loss = 0.0
            n_batches = 0
            
            pbar = tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            
            for i in pbar:
                x_batch = x_shuffled[i:i+batch_size]
                f_batch = f_shuffled[i:i+batch_size]
                
                def f_torch_batch(x):
                    if x.shape[0] != f_batch.shape[0]:
                        indices = torch.randint(0, f_batch.shape[0], (x.shape[0],))
                        return f_batch[indices]
                    return f_batch
                
                loss = zubov_loss(x_batch, self.lyapunov, f_torch_batch, self.device, mu=mu, transform=transform)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lyapunov.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=loss.item())
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
        return losses
    
def main():

    device = torch.device('cpu')

    nn_dynamics = NNDynamics(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        device=device
    )
    print(f" State dimension: {nn_dynamics.state_dim}")
    
    x_train, f_train = nn_dynamics.generate_collocation_points( num_trajectories=500, num_steps=15, learning_rate_range=(1e-3, 0.1))
    
    learner = LyapunovLearner(state_dim=nn_dynamics.state_dim, device=device)
    losses = learner.train(x_train, f_train, epochs=30, batch_size=128, mu=0.1, transform="exp")
    
    Path('models').mkdir(exist_ok=True)
    torch.save(learner.lyapunov.state_dict(), 'models/lyapunov_nn_training.pt')

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Lyapunov PINN Training Loss (NN Training Dynamics)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('lyapunov_training_loss.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()