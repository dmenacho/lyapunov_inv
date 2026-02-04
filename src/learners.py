import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def zubov_loss(
    x: torch.Tensor,
    V_net: nn.Module,
    f_batch: torch.Tensor,
    device: torch.device,
    mu: float = 0.1,
    transform: str = "exp",
) -> torch.Tensor:
    """
    Zubov residual-based loss adapted from LyZNet conventions:
      residual = V_dot + mu * ||x||^2 * (1 - V)     (exp transform)
    or:
      residual = V_dot + mu * ||x||^2 * (1 - V) * (1 + V) (other transform)
    plus origin constraints.
    """
    x = x.requires_grad_(True)
    V = V_net(x).squeeze()
    V_grad = torch.autograd.grad(outputs=V.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
    # Here f_batch corresponds to f(x) at the same batch points
    V_dot = (V_grad * f_batch).sum(dim=1)
    norm_sq = (x**2).sum(dim=1)

    if transform == "exp":
        residual = V_dot + mu * norm_sq * (1.0 - V)
    else:
        residual = V_dot + mu * norm_sq * (1.0 - V) * (1.0 + V)

    pde_loss = residual**2

    # Origin constraint: V(0) ~ 0 and grad V(0) ~ 0
    zero = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    zero = zero.requires_grad_(True)
    V0 = V_net(zero)
    V0_grad = torch.autograd.grad(outputs=V0.sum(), inputs=zero, create_graph=True)[0]
    origin_loss = (V0_grad**2).sum() + (V0**2).sum()

    return (pde_loss.mean() + origin_loss)

class DynamicSampler:
    def __init__(self, model_instance, device='cpu', num_train_samples=1000, data_dir=None):
        self.device = device
        #self.model = TinyResNet().to(device)
        self.model = model_instance.to(device)
        self.state_dim = self.model.count_parameters()
        self.data_dir = data_dir
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

    def simulate_trajectory(self, w_init: np.ndarray, learning_rate: float, num_steps: int, return_gradients: bool = True):
        model = self.model.to(self.device)
        self.unflatten_params(w_init, model)
        
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        trajectory = [w_init.copy()]
        f_list: List[np.ndarray] = []
        grad_list: List[np.ndarray] = []
        losses: List[np.ndarray] = []
        
        for step in range(num_steps):
            batch_idx = np.random.randint(0, len(self.X_train))
            X_batch = self.X_train[batch_idx]
            y_batch = self.y_train[batch_idx]
            
            model.zero_grad(set_to_none=True)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()

            grad_flat = torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
            f = (-grad_flat).cpu().numpy().astype(np.float32)  # f(w) = -∇L(w)

            if return_gradients:
                grad_list.append(grad_flat.cpu().numpy().astype(np.float32))
            f_list.append(f)

            with torch.no_grad():
                for param in model.parameters():
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
            
            w_current = self.flatten_params(model)
            trajectory.append(w_current.copy())
            losses.append(loss.cpu().detach().numpy())
        
        traj = np.asarray(trajectory, dtype=np.float32)  # (T+1, D)
        f_arr = np.asarray(f_list, dtype=np.float32)     # (T, D)
        f_arr = np.vstack([f_arr[0:1], f_arr])           # (T+1, D), repeat first at time 0

        grads = np.asarray(grad_list, dtype=np.float32) if return_gradients else None
        aux = (losses,)
        return traj, f_arr, grads, aux 
    
    def generate_collocation_points(self, num_trajectories: int,
                                    traj_steps: int,
                                    learning_rate: float,
                                    w_init_scale: float,
                                    normalize: bool = True,) :

        all_states = []
        all_velocities = []
        learning_rate_range=0.01
        for _ in tqdm(range(num_trajectories), desc="Generating trajectories"):
            w_init = (np.random.randn(self.state_dim).astype(np.float32) * w_init_scale)
            # lr = np.random.uniform(*learning_rate_range)
            
            trajectory, velocities, _, aux = self.simulate_trajectory(w_init, learning_rate, traj_steps, return_gradients=False)
            
            all_states.append(trajectory)
            all_velocities.append(velocities)
        
        x = np.vstack(all_states)
        f = np.vstack(all_velocities)
    
        stats: Dict[str, np.ndarray] = {}
        if normalize:
            x_mean = x.mean(axis=0)
            x_std = x.std(axis=0) + 1e-8
            f_mean = f.mean(axis=0)
            f_std = f.std(axis=0) + 1e-8

            x_n = (x - x_mean) / x_std
            f_n = (f - f_mean) / f_std

            stats = {"x_mean": x_mean, "x_std": x_std, "f_mean": f_mean, "f_std": f_std}
            x, f = x_n, f_n
        else:
            stats = {"x_mean": np.zeros(self.state_dim, dtype=np.float32),
                     "x_std": np.ones(self.state_dim, dtype=np.float32),
                     "f_mean": np.zeros(self.state_dim, dtype=np.float32),
                     "f_std": np.ones(self.state_dim, dtype=np.float32)}
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        f_tensor = torch.tensor(f, dtype=torch.float32, device=self.device)
        
        print(f"Generated {x_tensor.shape[0]} collocation points")
        
        return x_tensor, f_tensor, stats, aux

    def _load_dataset(self, num_samples):
        print("not implemented")
        pass


class ModelOCIFAR10Dynamics(DynamicSampler):
    def __init__(self, model_instance, device='cpu', num_train_samples=1000, data_dir=None):
        super().__init__(model_instance, device, num_train_samples, data_dir)

    #def _load_cifar10(self, num_samples):
    def _load_dataset(self, num_samples): # implemented with cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        #dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.dataset_train = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        self.dataset_test = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)
        
        indices = np.random.choice(len(self.dataset_train), num_samples, replace=False)
        subset = torch.utils.data.Subset(self.dataset_train, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
        
        self.X_train = []
        self.y_train = []
        
        for X, y in loader:
            self.X_train.append(X.to(self.device))
            self.y_train.append(y.to(self.device))


class ModelOMNISTDynamics(DynamicSampler):
    def __init__(self, model_instance, device='cpu', num_train_samples=1000, data_dir=None):
        super().__init__(model_instance, device, num_train_samples, data_dir)

    #def _load_cifar10(self, num_samples):
    def _load_dataset(self, num_samples: int) -> None:# implemented with cifar10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3),
                #transforms.Normalize((0.1307,0.1307,0.1307,), (0.3081,0.3081,0.3081,)),  # standard MNIST mean/std
            ]
        )

        t_transform = transforms.Compose(
            [
                transforms.ToTensor(),
        #        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST mean/std
            ]
        )
        self.dataset_train = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform, target_transform = t_transform)
        self.dataset_test = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform, target_transform = t_transform)

        num_samples = min(num_samples, len(self.dataset_train))
        indices = np.random.choice(len(self.dataset_train), num_samples, replace=False)

        subset = torch.utils.data.Subset(self.dataset_train, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)

        self.X_train = []
        self.y_train = []
        for X, y in loader:
            self.X_train.append(X.to(self.device))  # X shape: (B,1,28,28)
            self.y_train.append(y.to(self.device))
    

class LyapunovLearner:
    def __init__(self, lyap_model, optimizer, device: torch.device) -> None:
        self.device = device
        #self.state_dim = state_dim
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