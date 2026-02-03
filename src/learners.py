import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


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
    
     # Origin constraint: V(0) = 0
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero_tensor.requires_grad_(True)
    V_zero = V_net(zero_tensor)
    V_grad_zero = torch.autograd.grad(outputs=V_zero.sum(), inputs=zero_tensor, create_graph=True)[0]
    
    orign_loss =  (V_grad_zero**2).sum() + (V_zero**2)
    loss = (pde_loss + orign_loss).mean()
    return loss

class DynamicSampler:
    def __init__(self, model_instance, device='cpu', num_train_samples=1000):
        self.device = device
        #self.model = TinyResNet().to(device)
        self.model = model_instance.to(device)
        self.state_dim = self.model.count_parameters()
        self._load_dataset(num_train_samples)


    def flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params).astype(np.float32)
    
    def unflatten_params(self, flat_params, model):
        flat_params = np.asarray(flat_params).ravel()
        idx = 0
        for param in model.parameters():
            size = param.data.numel()
            param.data = torch.tensor(
                flat_params[idx:idx+size].reshape(param.shape),
                dtype=torch.float32,
                device=self.device
            )
            idx += size

    def simulate_trajectory(self, w_init, learning_rate, num_steps):
        model = self.model.to(self.device)
        self.unflatten_params(w_init, model)
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        trajectory = [w_init.copy()]
        
        for step in range(num_steps):
            batch_idx = np.random.randint(0, len(self.X_train))
            X_batch = self.X_train[batch_idx]
            y_batch = self.y_train[batch_idx]
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
            
            w_current = self.flatten_params(model)
            trajectory.append(w_current.copy())
        
        trajectory = np.array(trajectory)
        
        velocities = np.diff(trajectory, axis=0) * learning_rate
        # velocities = np.diff(trajectory, axis=0)
        velocities = np.vstack([velocities[0], velocities])

        aux = (loss.cpu().detach().numpy(),)
        
        return trajectory, velocities, aux
    
    def generate_collocation_points(self, num_trajectories=50, num_steps=10) :

        all_states = []
        all_velocities = []
        learning_rate_range=0.01
        for _ in tqdm(range(num_trajectories), desc="Generating trajectories"):
            w_init = np.random.randn(self.state_dim).astype(np.float32) * 0.05
            # lr = np.random.uniform(*learning_rate_range)
            
            trajectory, velocities, aux = self.simulate_trajectory(w_init, learning_rate_range, num_steps)
            
            all_states.append(trajectory)
            all_velocities.append(velocities)
        
        x_array = np.vstack(all_states)
        f_array = np.vstack(all_velocities)
    
        x_mean = x_array.mean(axis=0)
        x_std = x_array.std(axis=0) + 1e-8
        x_array = (x_array - x_mean) / x_std

        f_mean = f_array.mean(axis=0)
        f_std = f_array.std(axis=0) + 1e-8
        f_array = (f_array - f_mean) / f_std 
        
        x_tensor = torch.tensor(x_array, dtype=torch.float32, device=self.device)
        f_tensor = torch.tensor(f_array, dtype=torch.float32, device=self.device)
        
        print(f"Generated {x_tensor.shape[0]} collocation points")
        
        return x_tensor, f_tensor, aux

    def _load_dataset(self, num_samples):
        print("not implemented")
        pass


class ModelOCIFAR10Dynamics(DynamicSampler):
    def __init__(self, model_instance, device='cpu', num_train_samples=1000):
        super().__init__(model_instance, device, num_train_samples)

    #def _load_cifar10(self, num_samples):
    def _load_dataset(self, num_samples): # implemented with cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
        
        self.X_train = []
        self.y_train = []
        
        for X, y in loader:
            self.X_train.append(X.to(self.device))
            self.y_train.append(y.to(self.device))


class ModelOCIFAR10Dynamics(DynamicSampler):
    def __init__(self, model_instance, device='cpu', num_train_samples=1000):
        super().__init__(model_instance, device, num_train_samples)

    #def _load_cifar10(self, num_samples):
    def _load_mnist(self, num_samples: int) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST mean/std
            ]
        )
        dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)

        num_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)

        self.X_train = []
        self.y_train = []
        for X, y in loader:
            self.X_train.append(X.to(self.device))  # X shape: (B,1,28,28)
            self.y_train.append(y.to(self.device))


#class LyapunovLearner:
#    def __init__(self, lyap_model, optimizer , state_dim = 2, device= 'cpu'):
#        self.device = device
#        self.state_dim = state_dim
#        #self.lyapunov = LyapunovNet(state_dim=state_dim, hidden_dim=256).to(device)
#        self.lyapunov = lyap_model
#        #self.optimizer = optim.Adam(self.lyapunov.parameters(), lr=1e-3)
#        self.optimizer = optimizer
#    
#    def train(self, x_train, f_train, epochs=30, batch_size=126, mu=0.1, transform="exp"):
#        n_samples = x_train.shape[0]
#        losses = []
#        
#        for epoch in range(epochs):
#            perm = torch.randperm(n_samples)
#            x_shuffled = x_train[perm]
#            f_shuffled = f_train[perm]
#            
#            epoch_loss = 0.0
#            n_batches = 0
#            
#            pbar = tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
#            
#            for i in pbar:
#                x_batch = x_shuffled[i:i+batch_size]
#                f_batch = f_shuffled[i:i+batch_size]
#                
#                def f_torch_batch(x):
#                    if x.shape[0] != f_batch.shape[0]:
#                        indices = torch.randint(0, f_batch.shape[0], (x.shape[0],))
#                        return f_batch[indices]
#                    return f_batch
#                
#                loss = zubov_loss(x_batch, self.lyapunov, f_torch_batch, self.device, mu=mu, transform=transform)
#                
#                self.optimizer.zero_grad()
#                loss.backward()
#                torch.nn.utils.clip_grad_norm_(self.lyapunov.parameters(), max_norm=1.0)
#                self.optimizer.step()
#                
#                epoch_loss += loss.item()
#                n_batches += 1
#                pbar.set_postfix(loss=loss.item())
#            
#            avg_loss = epoch_loss / n_batches
#            losses.append(avg_loss)
#            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
#        
#        return losses
    

class LyapunovLearner:
    def __init__(self, state_dim: int, lyap_model, optimizer, device: torch.device) -> None:
        self.device = device
        self.state_dim = state_dim
        #self.lyapunov = LyapunovNet(state_dim=state_dim, hidden_dim=256).to(device)
        self.lyapunov = lyap_model
        #self.optimizer = optim.Adam(self.lyapunov.parameters(), lr=1e-3)
        self.optimizer = optimizer

    def train(
        self,
        x_train: torch.Tensor,
        f_train: torch.Tensor,
        epochs: int,
        batch_size: int,
        mu: float,
        transform: str,
        grad_clip: float = 1.0,
    ) -> List[float]:
        n = x_train.shape[0]
        losses: List[float] = []

        for epoch in range(epochs):
            perm = torch.randperm(n, device=x_train.device)
            x_shuffled = x_train[perm]
            f_shuffled = f_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(range(0, n, batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for i in pbar:
                xb = x_shuffled[i : i + batch_size]
                fb = f_shuffled[i : i + batch_size]

                self.optimizer.zero_grad(set_to_none=True)
                loss = zubov_loss(xb, self.lyapunov, fb, self.device, mu=mu, transform=transform)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lyapunov.parameters(), max_norm=grad_clip)
                self.optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1
                pbar.set_postfix(loss=float(loss.item()))

            avg = epoch_loss / max(1, n_batches)
            losses.append(avg)
            print(f"Epoch {epoch+1}/{epochs}: loss={avg:.6f}")

        return losses