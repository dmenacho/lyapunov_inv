import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms  

# -------------------------
# Model definitions
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class TinyResNet(nn.Module):

    def __init__(self, initial_dim, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(initial_dim, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.block1 = ResidualBlock(8, 8)
        self.block2 = ResidualBlock(8, 8)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LyapunovNet(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_compress = nn.Linear(state_dim, 128)
        self.net = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_compress(x)
        V = self.net(x)
        # V= torch.relu(V) + 1e-6
        V = V**2 + 1e-6
        return V

# -------------------------
# Dynamics (gradient descent on CIFAR-10 minibatches)
# -------------------------
class ResNetDynamics:
    """
    Implements:
      - flatten/unflatten of TinyResNet parameters
      - simulation of gradient descent trajectories on CIFAR-10
      - creation of collocation points (x, f(x)) for Lyapunov learning

    By default, we use the (discrete-time) gradient-descent vector field:
        f(w) := -∇_w L(w; minibatch)
    This matches the later stable-initialization search which can compute gradients.
    """

    def __init__(self, device: str = "cpu", num_train_samples: int = 1000, data_dir: str = "./data", data_type: str ="nmist") -> None:
        self.device = device
        self.data_dir = data_dir
        if data_type == "cifar":
            self._load_cifar10(num_train_samples)
            self.input_dim=3
        elif data_type == "nmist":
            self._load_mnist(num_train_samples)
            self.input_dim=1
        self.model = TinyResNet(self.input_dim).to(device)
        self.state_dim = self.model.count_parameters()

    def _load_cifar10(self, num_samples: int) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)

        self.X_train: List[torch.Tensor] = []
        self.y_train: List[torch.Tensor] = []
        for X, y in loader:
            self.X_train.append(X.to(self.device))
            self.y_train.append(y.to(self.device))
    
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

    @staticmethod
    def _flatten_params_torch(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in model.parameters()])

    def flatten_params(self, model: nn.Module) -> np.ndarray:
        return self._flatten_params_torch(model).detach().cpu().numpy().astype(np.float32)

    def unflatten_params(self, flat_params: Union[np.ndarray, torch.Tensor], model: nn.Module) -> None:
        if isinstance(flat_params, torch.Tensor):
            flat = flat_params.detach().cpu().numpy().ravel()
        else:
            flat = np.asarray(flat_params).ravel()
        idx = 0
        for p in model.parameters():
            size = p.data.numel()
            chunk = flat[idx : idx + size].reshape(p.shape)
            p.data = torch.tensor(chunk, dtype=torch.float32, device=self.device)
            idx += size

    def simulate_trajectory(
        self,
        w_init: np.ndarray,
        learning_rate: float,
        num_steps: int,
        return_gradients: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
          trajectory: (T+1, D) weights
          f:          (T+1, D) vector field samples at trajectory points (we repeat f_0 at index 0)
          gradients:  (T, D) raw gradients (optional)
        """
        model = TinyResNet(self.input_dim).to(self.device)
        self.unflatten_params(w_init, model)

        loss_fn = nn.CrossEntropyLoss()

        trajectory: List[np.ndarray] = [w_init.copy()]
        f_list: List[np.ndarray] = []
        grad_list: List[np.ndarray] = []

        for _ in range(num_steps):
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
                for p in model.parameters():
                    p.data -= learning_rate * p.grad
                    p.grad.zero_()

            w_current = self.flatten_params(model)
            trajectory.append(w_current.copy())

        traj = np.asarray(trajectory, dtype=np.float32)  # (T+1, D)
        f_arr = np.asarray(f_list, dtype=np.float32)     # (T, D)
        f_arr = np.vstack([f_arr[0:1], f_arr])           # (T+1, D), repeat first at time 0

        grads = np.asarray(grad_list, dtype=np.float32) if return_gradients else None
        return traj, f_arr, grads

    def generate_collocation_points(
        self,
        num_trajectories: int,
        traj_steps: int,
        learning_rate: float,
        w_init_scale: float,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
        """
        Returns:
          x_tensor: (N, D) states (weights)
          f_tensor: (N, D) vector field samples (normalized if requested)
          stats: {x_mean, x_std, f_mean, f_std}
        """
        all_states: List[np.ndarray] = []
        all_f: List[np.ndarray] = []

        for _ in tqdm(range(num_trajectories), desc="Generating trajectories"):
            w_init = (np.random.randn(self.state_dim).astype(np.float32) * w_init_scale)
            traj, f_arr, _ = self.simulate_trajectory(w_init, learning_rate, traj_steps, return_gradients=False)
            all_states.append(traj)
            all_f.append(f_arr)

        x = np.vstack(all_states)  # (num_trajectories*(T+1), D)
        f = np.vstack(all_f)       # same

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

        return x_tensor, f_tensor, stats


# -------------------------
# Zubov loss / Lyapunov learner
# -------------------------
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


class LyapunovLearner:
    def __init__(self, state_dim: int, device: torch.device, lr: float = 1e-3, hidden_dim: int = 256) -> None:
        self.device = device
        self.lyapunov = LyapunovNet(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.lyapunov.parameters(), lr=lr)

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


# -------------------------
# Stable initialization search
# -------------------------
class StabilityConstrainedNN:
    def __init__(
        self,
        lyapunov: LyapunovNet,
        resnet_dynamics: ResNetDynamics,
        learning_rate: float,
        traj_steps: int,
        device: torch.device,
        alpha: float = 0.05,
        constraint_threshold: float = 1.0,
    ) -> None:
        self.lyapunov = lyapunov.to(device)
        self.lyapunov.eval()
        self.nn_dynamics = resnet_dynamics
        self.device = device
        self.learning_rate = learning_rate
        self.traj_steps = traj_steps
        self.alpha = alpha
        self.constraint_threshold = constraint_threshold

        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.f_mean: Optional[torch.Tensor] = None
        self.f_std: Optional[torch.Tensor] = None

    def set_normalization_stats(self, x_mean, x_std, f_mean=None, f_std=None) -> None:
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32, device=self.device)
        self.x_std  = torch.as_tensor(x_std,  dtype=torch.float32, device=self.device)
        if f_mean is not None and f_std is not None:
            self.f_mean = torch.as_tensor(f_mean, dtype=torch.float32, device=self.device)
            self.f_std  = torch.as_tensor(f_std,  dtype=torch.float32, device=self.device)

    def simulate_training_trajectory_with_gradients(self, w_init: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (trajectory, gradients) where:
           trajectory: (T+1, D)
           gradients:  (T, D)
        """
        traj_np, _, grads_np = self.nn_dynamics.simulate_trajectory(
            w_init=w_init, learning_rate=self.learning_rate, num_steps=self.traj_steps, return_gradients=True
        )
        assert grads_np is not None
        traj = torch.tensor(traj_np, dtype=torch.float32, device=self.device)
        grads = torch.tensor(grads_np, dtype=torch.float32, device=self.device)
        return traj, grads

    def evaluate_lyapunov_on_trajectory(self, trajectory: torch.Tensor, gradients: torch.Tensor) -> Tuple[float, float]:
        """
        Approximate constraints:
          c1 = max(0, -V(w0))  (positivity)
          c2 = max_t max(0, dV/dt + alpha*V)  (exponential stability-type condition)
        where f(w) = -∇L(w) (matching training collocation).
        """
        if self.x_mean is not None and self.x_std is not None:
            traj_n = (trajectory - self.x_mean) / (self.x_std + 1e-8)
            grads_n = gradients / (self.x_std + 1e-8)
        else:
            # fallback (older checkpoints)
            traj_n = (trajectory - trajectory.mean(dim=0)) / (trajectory.std(dim=0) + 1e-8)
            grads_n = gradients / (trajectory.std(dim=0) + 1e-8)

        V_vals: List[float] = []
        Vdot_vals: List[float] = []

        for t in range(traj_n.shape[0] - 1):
            state = traj_n[t : t + 1].requires_grad_(True)
            V = self.lyapunov(state)  # (1,1)
            V_vals.append(float(V.item()))
            V_grad = torch.autograd.grad(V.sum(), state, create_graph=False)[0]  # (1,D)

            # grad_raw = grads_n[t]
            grad_raw = gradients[t] # this is ∇L
            f_raw = -grad_raw # f = -∇L

            if self.f_mean is not None and self.f_std is not None:
                f_n = (f_raw - self.f_mean) / (self.f_std + 1e-8)
            else:
                f_n = f_raw 
            Vdot = (V_grad.squeeze(0) * f_n).sum()
            Vdot_vals.append(float(Vdot.item()))

        V_vals_np = np.asarray(V_vals, dtype=np.float32)
        Vdot_np = np.asarray(Vdot_vals, dtype=np.float32)

        c1 = max(0.0, -float(V_vals_np[0]))
        stability = Vdot_np + self.alpha * V_vals_np
        c2 = float(np.maximum(stability, 0.0).max())
        constraint = c1 + c2
        return float(V_vals_np.max()), constraint

    def evaluate_initialization(self, w0: np.ndarray) -> Tuple[float, float]:
        traj, grads = self.simulate_training_trajectory_with_gradients(w0)
        return self.evaluate_lyapunov_on_trajectory(traj, grads)

    def discover_stable_initializations(
        self,
        n_samples: int,
        weight_scale_range: Tuple[float, float],
        verbose_every: int = 25,
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        stable_weights: List[np.ndarray] = []
        metadata: List[Dict[str, float]] = []

        scales = np.random.uniform(weight_scale_range[0], weight_scale_range[1], n_samples)
        for i, scale in enumerate(tqdm(scales, desc="Weight init sampling")):
            w0 = (np.random.randn(self.nn_dynamics.state_dim).astype(np.float32) * float(scale))

            try:
                Vmax, viol = self.evaluate_initialization(w0)

                if (i + 1) % max(1, verbose_every) == 0:
                    print(f"[{i+1}/{n_samples}] scale={scale:.4f}  Vmax={Vmax:.4f}  viol={viol:.4f}")

                if viol < self.constraint_threshold:
                    stable_weights.append(w0)
                    metadata.append(
                        {
                            "initial_scale": float(scale),
                            "V_max": float(Vmax),
                            "constraint_violation": float(viol),
                            "weight_norm": float(np.linalg.norm(w0)),
                        }
                    )
            except Exception as e:
                # keep going
                if (i + 1) % max(1, verbose_every) == 0:
                    print(f"[{i+1}/{n_samples}] error: {e}")
                continue

        return stable_weights, metadata


def analyze_stable_weights(stable_weights: List[np.ndarray], metadata: List[Dict[str, float]], state_dim: int) -> Dict[str, float]:
    if not stable_weights:
        print("No stable weights found.")
        return {}

    weight_norms = np.array([m["weight_norm"] for m in metadata], dtype=np.float32)
    Vmax = np.array([m["V_max"] for m in metadata], dtype=np.float32)
    viol = np.array([m["constraint_violation"] for m in metadata], dtype=np.float32)

    stats = {
        "count": int(len(stable_weights)),
        "weight_norm_mean": float(weight_norms.mean()),
        "weight_norm_std": float(weight_norms.std()),
        "Vmax_mean": float(Vmax.mean()),
        "Vmax_std": float(Vmax.std()),
        "viol_mean": float(viol.mean()),
        "viol_std": float(viol.std()),
    }

    print("\nStable init summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # quick peek at first few dimensions
    sw = np.array(stable_weights, dtype=np.float32)
    for d in range(min(10, state_dim)):
        col = sw[:, d]
        print(f"  w[{d}]: mean={col.mean():.6f}, std={col.std():.6f}, range=[{col.min():.6f}, {col.max():.6f}]")

    return stats


def plot_stable_weights(metadata: List[Dict[str, float]], out_path: Path, title_suffix: str = "") -> None:
    if not metadata:
        print("No metadata to plot.")
        return

    weight_norms = np.array([m["weight_norm"] for m in metadata], dtype=np.float32)
    Vmax = np.array([m["V_max"] for m in metadata], dtype=np.float32)
    viol = np.array([m["constraint_violation"] for m in metadata], dtype=np.float32)
    scales = np.array([m["initial_scale"] for m in metadata], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(weight_norms, bins=15, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Weight norm ||w0||")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Stable weight norms" + title_suffix)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(Vmax, viol, alpha=0.6, s=60)
    axes[0, 1].set_xlabel("V_max")
    axes[0, 1].set_ylabel("Constraint violation")
    axes[0, 1].set_title("V_max vs constraint" + title_suffix)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(scales, weight_norms, alpha=0.6, s=60)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Init scale")
    axes[1, 0].set_ylabel("Weight norm ||w0||")
    axes[1, 0].set_title("Scale -> norm" + title_suffix)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(viol, bins=15, alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Constraint violation")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Constraint distribution" + title_suffix)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")

# -------------------------
# Full pipeline runner
# -------------------------
def train_lyapunov_phase(args):
    """
    1) Build dynamics
    2) Generate collocation points
    3) Train LyapunovNet
    4) Save checkpoint + stats + losses
    Returns: (ckpt_path: Path, stats: dict, losses: list[float])
    """
    device = torch.device(args.device)

    dyn = ResNetDynamics(
        device=str(device),
        num_train_samples=args.num_train_samples,
        data_dir=args.data_dir,
        data_type = args.data_type
    )
    print(f"[train] state_dim={dyn.state_dim}")

    x_train, f_train, stats = dyn.generate_collocation_points(
        num_trajectories=args.num_trajectories,
        traj_steps=args.traj_steps,
        learning_rate=args.gd_lr,
        w_init_scale=args.w_init_scale,
        normalize=True,
    )

    learner = LyapunovLearner(
        state_dim=dyn.state_dim,
        device=device,
        lr=args.lyap_lr,
        hidden_dim=args.lyap_hidden_dim,
    )
    
    losses = learner.train(
        x_train=x_train,
        f_train=f_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mu=args.mu,
        transform=args.transform,
        grad_clip=args.grad_clip,
    )

    # Save checkpoint
    run_dir = Path(args.out_dir) / "models" / f"{args.num_trajectories}_{args.traj_steps}_{args.gd_lr}" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "lyapunov_checkpoint.pt"
    torch.save(
        {
            "lyapunov_state_dict": learner.lyapunov.state_dict(),
            "stats": {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()},
            "args": vars(args),
            "losses": losses,
        },
        ckpt_path,
    )
    ckpt_path_model = run_dir / "lyapunov_resnet.pt"
    torch.save(learner.lyapunov.state_dict(), ckpt_path_model)

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Lyapunov PINN Training Loss (ResNet Training Dynamics)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(run_dir/'lyapunov_resnet_training_loss.png', dpi=150, bbox_inches='tight')

    print(f"[train] saved ckpt: {ckpt_path}")
    return ckpt_path, stats, losses


def find_stable_phase(args, ckpt_path: Path):
    """
    1) Load Lyapunov checkpoint
    2) Create StabilityConstrainedNN
    3) Sample initializations and filter by constraint
    4) Save stable weights + metadata
    Returns: (stable_weights, stable_meta)
    """
    device = torch.device(args.device)

    # Build dynamics for the stable-search phase
    dyn = ResNetDynamics(
        device=str(device),
        num_train_samples=args.num_train_samples_search,
        data_dir=args.data_dir,
        data_type=args.data_type
    )
    print(f"[stable] state_dim={dyn.state_dim}")

    obj = torch.load(ckpt_path, map_location=device)
    lyap = LyapunovNet(state_dim=dyn.state_dim, hidden_dim=args.lyap_hidden_dim).to(device)
    lyap.load_state_dict(obj["lyapunov_state_dict"])
    stats = obj.get("stats", {})

    solver = StabilityConstrainedNN(
        lyapunov=lyap,
        resnet_dynamics=dyn,
        learning_rate=args.gd_lr,
        traj_steps=args.traj_steps,
        device=device,
        alpha=args.alpha,
        constraint_threshold=args.constraint_threshold,
    )
    if "x_mean" in stats and "x_std" in stats:
        solver.set_normalization_stats(stats["x_mean"], stats["x_std"], stats.get("f_mean", None), stats.get("f_std", None))
    else:
        print("[stable] warning: no x_mean/x_std in checkpoint; using fallback normalization.")

    stable_weights, stable_meta = solver.discover_stable_initializations(
        n_samples=args.search_samples,
        weight_scale_range=(args.weight_scale_min, args.weight_scale_max),
        verbose_every=args.verbose_every,
    )

    # Save artifacts
    results_dir = Path(args.out_dir) /  "results" / f"{args.num_trajectories}_{args.traj_steps}_{args.gd_lr}" /args.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(stable_weights) > 0:
        np.save(results_dir / "stable_weights.npy", np.asarray(stable_weights, dtype=np.float32))
        (results_dir / "stable_weights_metadata.json").write_text(
            json.dumps({"items": stable_meta}, indent=2)
        )
        plot_stable_weights(stable_meta, results_dir / "stable_weights_plots.png")
        print(f"[stable] saved stable results to: {results_dir}")
    else:
        print("[stable] no stable weights found")

    return stable_weights, stable_meta


def cmd_full(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # run id
    if not args.run_id:
        args.run_id = time.strftime("%Y%m%d_%H%M%S")

    # save args
    run_dir_params = Path(args.out_dir) / "params" / args.run_id
    run_dir_params.mkdir(parents=True, exist_ok=True)
    (run_dir_params / "params.json").write_text(json.dumps(vars(args), indent=2))

    # train -> stable
    ckpt_path, stats, losses = train_lyapunov_phase(args)
    stable_weights, stable_meta = find_stable_phase(args, ckpt_path)

    print(f"[full] stable_count={len(stable_weights)}")


# -------------------------
# CLI
# -------------------------
def build_parser():
    p = argparse.ArgumentParser("lyap_resnet_pipeline")

    # global
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./training_info")

    sub = p.add_subparsers(dest="command", required=True)

    # ONE command: full pipeline
    f = sub.add_parser("full", help="Run full pipeline: train Lyapunov then stable-search.")

    # --- train phase args ---
    f.add_argument("--num-train-samples", type=int, default=10000)
    f.add_argument("--num-trajectories", type=int, default=200)
    f.add_argument("--traj-steps", type=int, default=50)
    f.add_argument("--gd-lr", type=float, default=1e-2)
    f.add_argument("--data-type", type=str, default="cifar", choices=["cifar", "nmist"])

    f.add_argument("--w-init-scale", type=float, default=0.5)

    f.add_argument("--lyap-lr", type=float, default=1e-3)
    f.add_argument("--lyap-hidden-dim", type=int, default=256)
    f.add_argument("--epochs", type=int, default=100)
    f.add_argument("--batch-size", type=int, default=128)
    f.add_argument("--mu", type=float, default=0.1)
    f.add_argument("--transform", type=str, default="exp", choices=["exp", "poly"])
    f.add_argument("--grad-clip", type=float, default=1.0)

    # --- stable search phase args ---
    f.add_argument("--num-train-samples-search", type=int, default=10000, help="CIFAR samples used during stable-search dynamics.")
    f.add_argument("--search-samples", type=int, default=1000)
    f.add_argument("--weight-scale-min", type=float, default=0.1)
    f.add_argument("--weight-scale-max", type=float, default=1.0)
    f.add_argument("--alpha", type=float, default=0.05)
    f.add_argument("--constraint-threshold", type=float, default=1.0)
    f.add_argument("--verbose-every", type=int, default=5)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "full":
        cmd_full(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
